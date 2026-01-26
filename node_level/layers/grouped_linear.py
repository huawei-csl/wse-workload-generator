import logging

from node_level.common.tensor import Tensor
from utils import dtype_to_byte
from node_level.common.compute_graph import get_compute_graph


class GroupedLinear:
    '''
    Equivalent to n_groups Linear layers running in parallel. 
    For example: einsum(bshc,hcd->bshd), h is common in all terms, meaning the same computation is repeated for h times. Therefore, h: n_groups
    '''
    def __init__(self, uid, rank, n_groups, in_features, out_features, dtype) -> None:
        super().__init__()
        logging.debug("GroupedLinear layer {} with n_groups: {} weight dims: {} x {}".format(uid, n_groups, in_features, out_features))

        self.uid = uid 
        self.n_groups = n_groups
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.dtype = dtype

    def forward(self, x, stats=None):
        n_groups = x.dims[0]
        batch_dims = x.dims[1:-1]
        in_features = x.dims[-1]
        assert n_groups == self.n_groups, "Input dim0 {} does not match self.n_groups {}".format(n_groups, self.n_groups)
        assert in_features == self.in_features, "Input in_features {} does not match self.in_features {}".format(in_features, self.in_features)

        total_batch_dim = eval("*".join([str(d) for d in batch_dims]))

        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops(n_groups, total_batch_dim)
        hbm_reads = self.hbm_reads(n_groups)
        network_data = self.network_data()
        dims = self.get_dims(n_groups, batch_dims)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))

        out = Tensor(f"{self.uid}_out", self.rank, [n_groups] + batch_dims + [self.out_features])
        stats.append(self.uid, "GroupedLinear", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)
        return out

    def memory_footprint(self, *args):
        memory_footprint =  self.n_groups * self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return memory_footprint # weights only, in bytes
    
    def num_ops(self, n_groups, total_batch_dim):
        n_ops = n_groups * total_batch_dim * self.in_features * self.out_features
        return n_ops # in terms of number of MACs

    def hbm_reads(self, n_groups):
        rw = n_groups * self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return rw # weights only, in bytes

    def get_dims(self, n_groups, batch_dims):
        input_dims = [n_groups] + batch_dims + [self.in_features]
        weight_dims = [self.n_groups, self.in_features, self.out_features]
        out_dims = [self.n_groups] + batch_dims + [self.out_features]
        return str(input_dims) + " x " + str(weight_dims) + " -> " + str(out_dims)
    
    def network_data(self):
        return 0
