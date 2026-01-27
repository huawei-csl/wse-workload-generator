
import logging

from src.node_level.layers.linear import Linear
from src.node_level.layers.allreduce import Allreduce

from src.node_level.common.utils import intceil

class LMHead:
    def __init__(self, layer_id, hidden_size, vocab_size, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating LMHead layer {}".format(layer_id))

        self.uid = layer_id

        self.par_factor = dist_info.tp_attn * dist_info.sp
        self.comm_group = dist_info.dense_comm_groups["tp_dense"]

        vocab_size_per_device = intceil(vocab_size/self.par_factor)
        self.head = Linear(uid=f"{self.uid}_head", rank=dist_info.rank, in_features=hidden_size, out_features=vocab_size_per_device, dtype=dtype)

        if self.par_factor > 1:
            self.allreduce = Allreduce(self.uid+"_ar", dist_info.rank, hidden_size, self.comm_group, dtype)

    def forward(self, x, stats):
        x = self.head.forward(x, stats=stats)
        if self.par_factor > 1:
            self.allreduce.forward(x, stats=stats)

    def memory_footprint(self):
        mem_size = self.head.memory_footprint()
        return mem_size # in bytes