
import logging 

from src.node_level.common.compute_graph import get_compute_graph
from src.node_level.common.tensor import Tensor
from src.node_level.common.utils import dtype_to_byte

class Unicast:
    def __init__(self, uid, dims, src: int, dst: int, dtype) -> None:
        super().__init__()
        logging.debug("Unicast layer {} with vector size: {} src:{} dst:{}".format(uid, dims, src, dst))

        self.uid = uid
        self.dims = dims
        self.dtype = dtype
        self.src = src
        self.dst = dst

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        assert x.dims == self.dims, "Input vector size {} does not match expected vector size {}".format(x.dims, self.dims)
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims()

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        
        out = Tensor(f"{self.uid}_{self.src}", self.dst, (bsz, seqlen, hidden_dim))
        stats.append(self.uid, "Unicast", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.dst, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def get_dims(self):
        return self.dims
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self):
        vecsize = eval("*".join([str(d) for d in self.dims])) * dtype_to_byte(self.dtype) # a vec of this size is sent from a single source to multiple destionations
        logging.debug("{}: network data size: {} B".format(self.uid, vecsize))
        return vecsize # in bytes
