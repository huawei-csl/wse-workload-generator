
import logging

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
        
        # Create buffers for the whole allgather operation
        # For src -> dst, data is stored at out_buffs[src_id][dst_id]
        out_buffs = {src_id: {dst_id: Tensor(f"{self.uid}_ag_{src_id}", dst_id, (self.vector_sizes[j], seqlen, hidden_dim)) for i, dst_id in enumerate(self.dst_nodes)} for j, src_id in enumerate(self.dst_nodes)}

        dst_tensors = list(out_buffs[self.dist_info.rank].values())

        n_elem = x.numel()
        network_size = self.network_data(n_elem*dtype_to_byte(self.dtype))
        stats.append(self.uid, "AllGather", 0, 0, 0, network_size, comm_group=self.dst_nodes, dims=x.dims)
        get_compute_graph().add_node(self, [x], dst_tensors, attrs=None)

        # Return receive buffers corresponding to this node as destination
        return [out_buffs[src_id][self.dist_info.rank] for src_id in self.dst_nodes]

    def network_data(self, tensor_size):
        vecsize = 2 * tensor_size * (len(self.dst_nodes) - 1) # N-1 vec receive + N-1 vec send, N: no. of devices in a cluster
        logging.debug("network data size (send + receive): {} B".format(vecsize))
        return vecsize # in bytes

dist_manager = None
def get_dist_manager():
    global dist_manager
    if dist_manager is None:
        return DistManager()
    return dist_manager