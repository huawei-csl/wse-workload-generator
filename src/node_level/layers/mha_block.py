
import logging 

from src.node_level.layers.linear import Linear
from src.node_level.layers.allreduce import Allreduce
from src.node_level.layers.mha import SelfAttention

from src.node_level.common.utils import intceil

class GQABlock:
    def __init__(self, uid, hidden_size, num_attention_heads, num_key_value_heads, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating GQA block {}".format(uid))

        assert hidden_size % num_attention_heads == 0
        head_dim = hidden_size // num_attention_heads

        self.dist_info = dist_info

        num_heads_per_device = intceil(num_attention_heads / dist_info.tp_attn)
        num_kv_heads_per_device = intceil(num_key_value_heads / dist_info.tp_attn)

        self.uid = uid
        self.rank = dist_info.rank

        self.ops = {}
        self.ops["q_proj"] = Linear(uid+"_qproj", self.rank, hidden_size, num_heads_per_device * head_dim, dtype)
        self.ops["k_proj"] = Linear(uid+"_kproj", self.rank, hidden_size, num_kv_heads_per_device * head_dim, dtype)
        self.ops["v_proj"] = Linear(uid+"_vproj", self.rank, hidden_size, num_kv_heads_per_device * head_dim, dtype)

        self.ops["self_attn"] = SelfAttention(uid+"_selfattn", num_heads_per_device, num_kv_heads_per_device, head_dim, dist_info.sp, dtype)
        if dist_info.sp > 1:
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", self.rank, num_heads_per_device * head_dim, dist_info.attn_comm_groups["sp"], dtype)

        self.ops["o_proj"] = Linear(uid+"_oproj", self.rank, num_heads_per_device * head_dim, hidden_size, dtype)

        if dist_info.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", self.rank, hidden_size, dist_info.attn_comm_groups["tp_attn"], dtype)

    def forward(self, bsz, seqlen, ctx_len, stats):
        batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        local_bsz = len(batch_ids)

        for opname in self.ops:
            if isinstance(self.ops[opname], SelfAttention):
                self.ops[opname].forward(local_bsz, seqlen, ctx_len, stats=stats)
            else:
                self.ops[opname].forward(local_bsz*seqlen, stats=stats)
                
    def memory_footprint(self, bsz, ctx_len):
        batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        local_bsz = len(batch_ids)

        mem_size = sum([self.ops[opname].memory_footprint(local_bsz, ctx_len) for opname in self.ops])
        return mem_size # in bytes

