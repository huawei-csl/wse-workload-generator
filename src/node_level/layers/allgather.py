
import logging 

from src.node_level.common.compute_graph import get_compute_graph
from src.node_level.common.tensor import Tensor
from src.node_level.common.utils import dtype_to_byte

class AllGather:
    def __init__(self, uid, vector_size, cluster_size, dist_info, dtype) -> None:
        super().__init__()
        # raise NotImplementedError("Not yet implemented, ask for support")
    
        logging.debug("AllGather layer {} with vector size: {} among {} devices".format(uid, vector_size, cluster_size))

        self.uid = uid
        self.vector_size = vector_size
        self.cluster_size = cluster_size
        self.dtype = dtype
        self.dist_info = dist_info
        self.comm_group = list(range(cluster_size))

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data(bsz*seqlen)
        dims = self.get_dims(bsz*seqlen)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        outs = [Tensor(f"{x.uid}_ag_{self.dist_info.rank}", dst_id, (bsz, seqlen, hidden_dim)) for dst_id in self.comm_group]
        stats.append(self.uid, "AllGather", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.comm_group, dims=dims)
        get_compute_graph().add_node(self, [x], outs, attrs=None)
        return outs[self.dist_info.rank]
    
    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def get_dims(self, bsz):
        vec_dims = [bsz, self.vector_size]
        return str(vec_dims)
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self, bsz):
        vecsize = 2 * bsz * self.vector_size * (self.cluster_size - 1) * dtype_to_byte(self.dtype) # N-1 vec receive + N-1 vec send, N: no. of devices in a cluster
        logging.debug("{}: network data size (send + receive): {} B".format(self.uid, vecsize))
        return vecsize # in bytes
