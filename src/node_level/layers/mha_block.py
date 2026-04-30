
import logging 

from src.node_level.layers.linear import Linear
from src.node_level.layers.allreduce import Allreduce
from src.node_level.layers.mha import SelfAttention

from src.node_level.common.stats import NodeStats
from src.node_level.common.utils import intceil
from src.node_level.common.tensor import View

class GQABlock:
    def __init__(self, uid, hidden_size, num_attention_heads, num_key_value_heads, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating GQA block {}".format(uid))

        assert hidden_size % num_attention_heads == 0
        head_dim = hidden_size // num_attention_heads
        self.head_dim = head_dim

        self.dist_info = dist_info

        num_heads_per_device = intceil(num_attention_heads / dist_info.tp_attn)
        self.num_heads_per_device = num_heads_per_device

        num_kv_heads_per_device = intceil(num_key_value_heads / dist_info.tp_attn)
        self.num_kv_heads_per_device = num_kv_heads_per_device

        self.uid = uid
        self.rank = dist_info.rank

        self.ops = {}
        self.ops["q_proj"] = Linear(uid+"_qproj", self.rank, hidden_size, num_heads_per_device * head_dim, dtype)
        self.ops["k_proj"] = Linear(uid+"_kproj", self.rank, hidden_size, num_kv_heads_per_device * head_dim, dtype)
        self.ops["v_proj"] = Linear(uid+"_vproj", self.rank, hidden_size, num_kv_heads_per_device * head_dim, dtype)

        self.ops["self_attn"] = SelfAttention(uid+"_selfattn", self.rank, num_heads_per_device, num_kv_heads_per_device, head_dim, dist_info.sp, dtype)
        if dist_info.sp > 1:
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", self.rank, num_heads_per_device * head_dim, dist_info.attn_comm_groups["sp"], dtype)

        self.ops["o_proj"] = Linear(uid+"_oproj", self.rank, num_heads_per_device * head_dim, hidden_size, dtype)

        if dist_info.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", self.rank, hidden_size, dist_info.attn_comm_groups["tp_attn"], dtype)

        self._stats = NodeStats()

    def forward(self, x, ctx_len, stats):
        self._stats.new_iter(stats.iter)

        local_bsz, seqlen, _ = x.dims # input x is a minibatch based on a given dp factor

        q = self.ops["q_proj"].forward(x, stats=self._stats)
        k = self.ops["k_proj"].forward(x, stats=self._stats)
        v = self.ops["v_proj"].forward(x, stats=self._stats)

        q = View(q, [local_bsz, seqlen, self.num_heads_per_device, self.head_dim]).forward(stats=self._stats)
        attn_out = self.ops["self_attn"].forward(q, ctx_len, stats=self._stats)
        attn_out = View(attn_out, [local_bsz, seqlen, self.num_heads_per_device*self.head_dim]).forward(stats=self._stats)

        if self.dist_info.sp > 1: 
            attn_out = self.ops["allreduce_sp"].forward(attn_out, stats=self._stats)

        y = self.ops["o_proj"].forward(attn_out, stats=self._stats)
        if self.dist_info.tp_attn > 1:
            y = self.ops["allreduce_tp"].forward(y, stats=self._stats)
        
        stats.merge(self._stats)

        return y 

    def memory_footprint(self, bsz, ctx_len):
        batch_ids = self.dist_info.get_local_batchids("attn")
        local_bsz = len(batch_ids)

        mem_size = sum([self.ops[opname].memory_footprint(local_bsz, ctx_len) for opname in self.ops])
        return mem_size # in bytes

