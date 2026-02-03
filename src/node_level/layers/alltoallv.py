import logging 

from src.node_level.common.compute_graph import get_compute_graph
from src.node_level.common.tensor import Tensor
from src.node_level.common.utils import dtype_to_byte

class AllToAllv:
    def __init__(self, uid, rank, input_split, output_split, comm_group, dtype) -> None:
        super().__init__()
        logging.debug("AllToAllv layer: {} with input_split {} and output_split {}".format(uid, input_split, output_split))

        self.axis = 0 # currently only support axis=0

        assert len(comm_group) > 0, "Communication group cannot be empty"
        assert len(input_split) == len(output_split) == len(comm_group), "Input split, output split, and comm group must have the same length"
        assert rank in comm_group, "Rank must be in the communication group"
        assert input_split[rank] == output_split[rank], "Input split and output split for the local rank must be the same"

        self.uid = uid
        self.rank = rank
        self.input_split = input_split
        self.output_split = output_split
        self.dtype = dtype
        self.comm_group = comm_group
        
    def forward(self, inputs, stats=None):
        assert len(inputs) == len(self.comm_group), "Number of input tensors must match the size of the communication group"
        assert sum([x.dims[0] for x in inputs]) == sum(self.input_split), "Sum of input tensor splits must match sum of input_split"

        input_dims = list(inputs[0].dims)
        input_dims[self.axis] = sum(self.input_split)

        output_dims = list(inputs[0].dims)
        output_dims[self.axis] = sum(self.output_split)

        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data(input_dims, output_dims)
        
        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, input_dims))
        
        outputs = []
        for dst_node_id in self.comm_group:
            out_buff_dim = list(output_dims)
            out_buff_dim[self.axis] = self.output_split[dst_node_id]
            outputs.append(Tensor(f"{self.uid}_{self.rank}", dst_node_id, out_buff_dim))
        
        stats.append(self.uid, "AlltoAll", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.comm_group, dims=f"{input_dims} -> {self.input_split} -> {self.output_split}")
        get_compute_graph().add_node(self, inputs, outputs, attrs=None)
        return outputs
    
    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self, input_dims, output_dims):
        input_size = eval("*".join([str(d) for d in input_dims])) * dtype_to_byte(self.dtype)
        output_size = eval("*".join([str(d) for d in output_dims])) * dtype_to_byte(self.dtype)
        network_size = input_size + output_size # 1 vec receive + 1 vec send
        logging.debug("{}: network data size (send + receive): {} B".format(self.uid, network_size))
        return network_size # in bytes
