
import logging

from src.node_level.common.tensor import Tensor
from src.node_level.common.utils import dtype_to_byte, intceil
from src.node_level.common.compute_graph import get_compute_graph

class Allreduce:
    ''' Allreduce Layer.
    Args:
        uid: unique identifier for the layer
        dims: dimensions of input tensor
        comm_group: communication group for allreduce
        dist_info: distributed information object
        dtype: data type of the tensor (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, rank, dims, comm_group, dtype) -> None:
        super().__init__()
        logging.debug("Allreduce layer {} with vector size: {} ".format(uid, dims))

        self.uid = uid
        self.dtype = dtype
        self.comm_group = comm_group
        self.rank = rank

    def forward(self, x, stats=None):
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data(x.dims)
        dims = self.get_dims(x.dims)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        
        out = Tensor(f"{self.uid}_out", self.rank, x.dims)
        stats.append(self.uid, "AllReduce", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.comm_group, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)
        return out

    def memory_footprint(self, *args):
        return 0

    def get_dims(self, dims):
        return str(dims)
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self, dims):
        vector_size = eval("*".join([str(d) for d in dims]))
        chunk_size = intceil(vector_size / len(self.comm_group)) * dtype_to_byte(self.dtype)
        traffic_per_node_per_round = 2 * chunk_size # each node sends and receives a chunk in each round
        num_rounds = len(self.comm_group) - 1
        traffic_per_node = traffic_per_node_per_round * num_rounds

        total_traffic = 2 * traffic_per_node # one full round for reduce and one full round for gather

        logging.debug("{}: network data size (send + receive) per node: {} B".format(self.uid, total_traffic))
        return total_traffic # in bytes
