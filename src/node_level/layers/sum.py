
import logging

from src.node_level.common.compute_graph import get_compute_graph
from src.node_level.common.tensor import Tensor

class Sum:
    def __init__(self, uid, dims, axis, dist_info, dtype) -> None:
        super().__init__()
        logging.debug("Sum layer {} with dims: {}".format(uid, dims))

        self.uid = uid 
        self.dims = dims
        self.axis = axis

        self.out_dims = list(self.dims)
        self.out_dims[self.axis] = 1

        self.dist_info = dist_info
        self.dtype = dtype

    def forward(self, x, stats=None):
        assert self.dims == x.dims, "Input dims {} does not match layer dims {}".format(x.dims, self.dims)

        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims()

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))

        out = Tensor(f"{self.uid}_out", self.dist_info.rank, self.out_dims) # squeeze seqlen
        stats.append(self.uid, "Sum", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0
    
    def get_dims(self):
        return str(self.dims) + " -> " + str(self.axis) + " -> " + str(self.out_dims)
    
    def num_ops(self):
        # n_ops = eval("*".join([str(d) for d in self.dims]))
        n_ops = 0
        return n_ops # in terms of number of MACs

    def hbm_reads(self):
        # rw = eval("*".join([str(d) for d in self.dims])) * dtype_to_byte(self.dtype)
        rw = 0
        return rw # weights only, in bytes

    def network_data(self):
        return 0
