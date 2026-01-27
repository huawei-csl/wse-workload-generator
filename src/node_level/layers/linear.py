import logging

from src.node_level.common.tensor import Tensor
from src.node_level.common.utils import dtype_to_byte
from src.node_level.common.compute_graph import get_compute_graph

class Linear:
    ''' Linear Layer.
    Args:
        uid: unique identifier for the layer
        rank: rank of the node where this layer is located
        in_features: size of each input sample
        out_features: size of each output sample
        dtype: data type of the weights (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, rank, in_features, out_features, dtype) -> None:
        super().__init__()
        logging.debug("Linear layer {} with weight dims: {} x {}".format(uid, in_features, out_features))

        self.uid = uid 
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

    def forward(self, x, stats=None):
        batch_dims = x.dims[:-1]
        total_batch_dim = eval("*".join([str(d) for d in batch_dims]))
        hidden_dim = x.dims[-1]
        assert hidden_dim == self.in_features, "Input hidden dim {} does not match in_features {}".format(hidden_dim, self.in_features)
        out_dim = list(x.dims)
        out_dim[-1] = self.out_features

        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops(total_batch_dim)
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims(total_batch_dim)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        out = Tensor(f"{self.uid}_out", self.rank, out_dim) # squeeze seqlen
        stats.append(self.uid, "Linear", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, *args):
        memory_footprint =  self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return memory_footprint # weights only, in bytes
    
    def get_dims(self, total_batch_dim):
        input_dims = [total_batch_dim, self.in_features]
        weight_dims = [self.in_features, self.out_features]
        out_dims = [total_batch_dim, self.out_features]
        return str(input_dims) + " x " + str(weight_dims) + " -> " + str(out_dims)
    
    def num_ops(self, total_batch_dim):
        n_ops = total_batch_dim * self.in_features * self.out_features
        return n_ops # in terms of number of MACs

    def hbm_reads(self):
        rw = self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return rw # weights only, in bytes

    def network_data(self):
        return 0