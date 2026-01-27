import logging
from src.node_level.common.tensor import Tensor
from src.node_level.common.compute_graph import get_compute_graph

class Add:
    ''' Element-wise Addition Layer.
    Args:
        uid: unique identifier for the layer
        rank: rank of the node where this layer is located
        dims: dimensions of input tensors
        dtype: data type of the tensors (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, rank, dims, dtype) -> None:
        super().__init__()
        logging.debug("Add layer {} with dims: {}".format(uid, dims))

        self.uid = uid
        self.rank = rank
        self.dims = dims
        self.dtype = dtype
    
    def forward(self, x0, x1, stats=None):
        assert x0.dims == self.dims, "Input tensor x0 dims do not match layer dims"
        assert x0.dims == x1.dims, "Input tensors must have the same dimensions for addition"

        out_dim = list(x0.dims)

        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims()

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        out = Tensor(f"{self.uid}_out", self.rank, out_dim) # squeeze seqlen
        stats.append(self.uid, "Add", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x0, x1], [out], attrs=None)
        return out

    def memory_footprint(self, *args):
        return 0 # no weights or KV-cache to store
    
    def get_dims(self):
        return str(self.dims) 
    
    def num_ops(self):
        return 0 # this is element-wise addition and does not use CUBE units

    def hbm_reads(self):
        return 0

    def network_data(self):
        return 0