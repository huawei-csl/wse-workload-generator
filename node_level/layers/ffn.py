import logging

from node_level.layers.linear import Linear
from node_level.layers.allreduce import Allreduce
from node_level.layers.add import Add
from stats import NodeStats

from utils import intceil

# Each expert is an instance of this layer
class FFN:
    def __init__(self, uid, hidden_size, intermediate_size, dist_info, dtype, is_dense_layer=False) -> None:
        super().__init__()
        logging.info("Creating FFN layer {}".format(uid))
        self.uid = uid
        self.rank = dist_info.rank
        self.dist_info = dist_info
        self.dtype = dtype
        
        if is_dense_layer:
            self.par_factor = dist_info.ep
            self.comm_group = dist_info.ffn_comm_groups["ep"]
        else:
            self.par_factor = dist_info.tp_ffn
            self.comm_group = dist_info.ffn_comm_groups["tp_ffn"]

        inter_size_per_node = intceil(intermediate_size/self.par_factor) 

        self.ops = {}
        self.ops["up"] = Linear(uid+"_up", self.rank, hidden_size, inter_size_per_node, dtype)
        self.ops["gate"] = Linear(uid+"_gate", self.rank, hidden_size, inter_size_per_node, dtype)
        self.ops["down"] = Linear(uid+"_down", self.rank, inter_size_per_node, hidden_size, dtype)
        
        # in case we use TP for experts
        if self.par_factor > 1:
            self.ops["allreduce"] = Allreduce(uid+"_ar", self.rank, hidden_size, self.comm_group, dtype)

        self._stats = NodeStats()

    def forward(self, x, stats):
        self._stats.new_iter(stats.iter)

        x1 = self.ops["up"].forward(x, stats=self._stats)
        x2 = self.ops["gate"].forward(x, stats=self._stats)
        
        x_add = Add(self.uid+"_add", self.rank, x1.dims, dtype=self.dtype).forward(x1, x2, stats=self._stats)

        y = self.ops["down"].forward(x_add, stats=self._stats)

        if self.par_factor > 1:
            y = self.ops["allreduce"].forward(y, stats=self._stats)
        
        stats.merge(self._stats)
        return y

    def memory_footprint(self, bsz=None, ctx_len=None):
        mem_size = sum([self.ops[opname].memory_footprint() for opname in self.ops])
        return mem_size # in bytes
