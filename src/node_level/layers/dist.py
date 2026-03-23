
import logging
import time 

from src.node_level.common.tensor import Tensor
from src.node_level.common.compute_graph import get_compute_graph
from src.node_level.common.utils import dtype_to_byte

class DistManager:
    def __init__(self) -> None:
        self.dist_ops = {}


    def register_op(self, op):
        if op.uid not in self.dist_ops:
            self.dist_ops[op.uid] = op
        else:
            assert self.dist_ops[op.uid]["vector_sizes"] == op.vector_sizes, "Vector sizes must be the same for the same uid"
            assert self.dist_ops[op.uid]["dst_nodes"] == op.dst_nodes, "Destination nodes must be the same for the same uid"

    def allgather(self, uid, x, vector_sizes, dst_nodes, dist_info, dtype, stats=None):
        op = AllGather(uid, vector_sizes, dst_nodes, dist_info, dtype)
        self.register_op(op)
        return op.forward(x, stats)

    def alltoallv(self, uid, x, send_split, recv_split, comm_group, dist_info, dtype, stats=None):
        op = AllToAllv(uid, dist_info.rank, send_split, recv_split, comm_group, dtype)
        self.register_op(op)
        return op.forward(x, stats)
    
class AllGather:
    def __init__(self, uid, vector_sizes, dst_nodes, dist_info, dtype) -> None:
        super().__init__()
    
        logging.debug("AllGather layer {} with vector size: {} among nodes {}".format(uid, vector_sizes, dst_nodes))

        self.uid = uid
        self.vector_sizes = vector_sizes
        self.dst_nodes = dst_nodes
        self.dtype = dtype
        self.dist_info = dist_info

        assert len(vector_sizes) == len(dst_nodes), "Length of vector_sizes and dst_nodes must be the same"
        assert dist_info.rank in dst_nodes, "Current rank must be in the list of destination nodes"

    def forward(self, x, stats=None):
        assert len(x.dims) == 3, "Input tensor must be 3D with shape [bsz, seqlen, hidden_dim]"
        _, seqlen, hidden_dim = x.dims

        assert x.dims[0] == self.vector_sizes[self.dist_info.rank]
        
        # Receive buffers. recv_buffs[i] corresponds to the buffer for receiving data from dst_nodes[i]
        recv_buffs = [Tensor(f"{self.uid}_ag_{src_id}", self.dist_info.rank, (self.vector_sizes[j], seqlen, hidden_dim)) for j, src_id in enumerate(self.dst_nodes)]
        
        # Destination tensors. dst_tensors[i] corresponds to the tensor for sending data to dst_nodes[i]
        dst_buffs = [Tensor(f"{self.uid}_ag_{self.dist_info.rank}", dst_id, (self.vector_sizes[self.dist_info.rank], seqlen, hidden_dim)) for i, dst_id in enumerate(self.dst_nodes)]

        n_elem = x.dims[0] * x.dims[1] * x.dims[2]
        network_size = self.network_data(n_elem*dtype_to_byte(self.dtype))
        stats.append(self.uid, "AllGather", 0, 0, 0, network_size, comm_group=self.dst_nodes, dims=x.dims)

        get_compute_graph().add_node(self, [x], dst_buffs, attrs=None)

        # Return receive buffers corresponding to this node as destination
        return recv_buffs

    def network_data(self, tensor_size):
        vecsize = tensor_size * (len(self.dst_nodes) - 1) # N-1 vec send, N: no. of devices in a cluster
        logging.debug("network data size (send + receive): {} B".format(vecsize))
        return vecsize # in bytes


class AllToAllv:
    def __init__(self, uid, rank, send_split, recv_split, comm_group, dtype) -> None:
        super().__init__()
        logging.debug("AllToAllv layer: {} with send_split {} and recv_split {}".format(uid, send_split, recv_split))

        self.axis = 0 # currently only support axis=0

        assert len(comm_group) > 0, "Communication group cannot be empty"
        assert len(send_split) == len(recv_split) == len(comm_group), "Input split, output split, and comm group must have the same length"
        assert rank in comm_group, "Rank must be in the communication group"
        assert send_split[rank] == recv_split[rank], "Input split and output split for the local rank must be the same"

        self.uid = uid
        self.rank = rank
        self.send_split = send_split
        self.recv_split = recv_split
        self.dtype = dtype
        self.comm_group = comm_group
        
    def forward(self, x, stats=None):
        assert len(x) == len(self.comm_group), "Number of input tensors must match the size of the communication group"
        assert sum([_x.dims[0] for _x in x]) == sum(self.send_split), "Sum of input tensor splits must match sum of send_split"

        for _x in x:
            assert _x.dims[1:] == x[0].dims[1:], "All input tensors must have the same dimensions except for the split dimension"

        input_dims = list(x[0].dims)
        input_dims[self.axis] = sum([s for i, s in enumerate(self.send_split) if self.rank != i])

        output_dims = list(x[0].dims)
        output_dims[self.axis] = sum([s for i, s in enumerate(self.recv_split) if self.rank != i])

        network_data = self.network_data(input_dims, output_dims)
        
        logging.debug("{} network data: {} B, dims: {}".format(self.uid, network_data, input_dims))
        
        dst_buffs = []
        for dst_node_id in self.comm_group:
            dst_buff_dim = list(output_dims)
            dst_buff_dim[self.axis] = self.send_split[dst_node_id]
            dst_buffs.append(Tensor(f"{self.uid}_{self.rank}", dst_node_id, dst_buff_dim))
        
        recv_buffs = []
        for src_node_id in self.comm_group:
            recv_buff_dim = list(input_dims)
            recv_buff_dim[self.axis] = self.recv_split[src_node_id]
            recv_buffs.append(Tensor(f"{self.uid}_{src_node_id}", self.rank, recv_buff_dim))

        stats.append(self.uid, "AlltoAll", 0, 0, 0, network_data, comm_group=self.comm_group, dims=f"{input_dims} -> {self.send_split} -> {self.recv_split}")
        get_compute_graph().add_node(self, x, dst_buffs, attrs=None)
        return recv_buffs
    
    def network_data(self, input_dims, output_dims):
        input_size = eval("*".join([str(d) for d in input_dims])) * dtype_to_byte(self.dtype)
        # output_size = eval("*".join([str(d) for d in output_dims])) * dtype_to_byte(self.dtype)
        network_size = input_size # total data size that is sent from this node
        logging.debug("{}: network data size (send + receive): {} B".format(self.uid, network_size))
        return network_size # in bytes



dist_manager = None
def get_dist_manager():
    global dist_manager
    if dist_manager is None:
        return DistManager()
    return dist_manager