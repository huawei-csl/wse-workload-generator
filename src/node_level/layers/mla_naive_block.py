
import logging

from src.node_level.layers.linear import Linear
from src.node_level.layers.mla_naive import MLANaiveAttention
from src.node_level.layers.allreduce import Allreduce
from src.node_level.common.stats import NodeStats
from src.node_level.common.tensor import Tensor, View, Split, Concat, Slice, Transpose

from src.node_level.common.utils import dtype_to_byte, intceil

class MLANaiveBlock:
    def __init__(self, uid, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype) -> None:
        super().__init__()
        logging.info("Creating MLA naive layer {}".format(uid))

        self.uid = uid
        self.dist_info = dist_info
        self.next_layer = next_layer
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.dtype = dtype
        self.rank = dist_info.rank

        n_local_heads = intceil(n_heads / dist_info.tp_attn)
        self.n_local_heads = n_local_heads

        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.qk_head_dim = qk_head_dim

        self.ops = {}
        self.ops["wq_a"] = Linear(uid+"_wqa", self.rank, hidden_size, q_lora_rank, dtype)
        self.ops["wq_b"] = Linear(uid+"_wqb", self.rank, q_lora_rank, n_local_heads*qk_head_dim, dtype)
        self.ops["wkv_a"] = Linear(uid+"_wkva", self.rank, hidden_size, kv_lora_rank+qk_rope_head_dim, dtype)
        self.ops["wkv_b"] = Linear(uid+"_wkvb", self.rank, kv_lora_rank, n_local_heads*(qk_nope_head_dim + v_head_dim), dtype)

        self.ops["naive_attn"] = MLANaiveAttention(uid+"_naiveattn", self.rank, n_local_heads, qk_head_dim, v_head_dim, dist_info.sp, dtype)
        if dist_info.sp > 1:
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", self.rank, n_local_heads * v_head_dim, dist_info.attn_comm_groups["sp"], dtype)

        self.ops["wo"] = Linear(uid+"_wo", self.rank, n_local_heads*v_head_dim, hidden_size, dtype)
        if dist_info.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", self.rank, hidden_size, dist_info.attn_comm_groups["tp_attn"], dtype)

        self._stats = NodeStats()

    def forward(self, x, ctx_len, stats):    
        self._stats.new_iter(stats.iter)

        local_bsz, seqlen, _ = x.dims # input x is a minibatch based on a given dp factor

        assert ctx_len == 0, "Naive block should be used only for prefill"

        kv_a = self.ops["wkv_a"].forward(x, stats=self._stats)
        kv_nope, kv_pe = Split(kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1).forward(stats=self._stats)
        
        kv_nope = self.ops["wkv_b"].forward(kv_nope, stats=self._stats)
        kv_nope = View(kv_nope, [local_bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim]).forward(stats=self._stats)

        kv_nope, v_cache = Split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], axis=-1).forward(stats=self._stats)

        kv_pe = View(kv_pe, [local_bsz, seqlen, 1, self.qk_rope_head_dim]).forward(stats=self._stats) 
        kv_pe_expand = Concat([kv_pe for i in range(self.n_local_heads)], axis=-2).forward(stats=self._stats)

        k_cache = Concat([kv_pe_expand, kv_nope], axis=-1).forward(stats=self._stats)

        q = self.ops["wq_a"].forward(x, stats=self._stats)
        q = self.ops["wq_b"].forward(q, stats=self._stats)
        q = View(q, [local_bsz, seqlen, self.n_local_heads, self.qk_head_dim]).forward(stats=self._stats)
        
        attn_out = self.ops["naive_attn"].forward(q, ctx_len, stats=self._stats)
        attn_out = View(attn_out, [local_bsz, seqlen, self.n_local_heads*self.v_head_dim]).forward(stats=self._stats)

        if self.dist_info.sp > 1:
            attn_out = self.ops["allreduce_sp"].forward(attn_out, stats=self._stats)

        y = self.ops["wo"].forward(attn_out, stats=self._stats)

        if self.dist_info.tp_attn > 1:
            y = self.ops["allreduce_tp"].forward(y, stats=self._stats)
        
        stats.merge(self._stats)

        return y

    def memory_footprint(self, bsz, ctx_len):
        batch_ids = self.dist_info.get_local_batchids("attn")
        local_bsz = len(batch_ids)
    
        mem_size = sum([self.ops[opname].memory_footprint(local_bsz, ctx_len) for opname in self.ops])
        return mem_size # in bytes

    def calc_expected(self, local_bsz, seqlen, ctx_len):
        assert ctx_len == 0, "Naive block should be used only for prefill"

        local_seqlen = intceil(seqlen / self.dist_info.sp)

        memory_footprint = self.hidden_size * self.q_lora_rank * dtype_to_byte(self.dtype) # wq_a
        memory_footprint += self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_to_byte(self.dtype) # wkv_a
        memory_footprint += self.q_lora_rank * self.n_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim) * dtype_to_byte(self.dtype) # wq_b
        memory_footprint += self.n_local_heads * self.kv_lora_rank * (self.qk_nope_head_dim + self.v_head_dim) * dtype_to_byte(self.dtype) # wkv_b
        memory_footprint += self.n_local_heads * self.v_head_dim * self.hidden_size * dtype_to_byte(self.dtype) # wo

        num_ops = local_bsz * seqlen * self.hidden_size * self.q_lora_rank # wq_a
        num_ops += local_bsz * seqlen * self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim) # wkv_a
        num_ops += local_bsz * seqlen * self.q_lora_rank * self.n_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim) # wq_b
        num_ops += local_bsz * seqlen * self.n_local_heads * self.kv_lora_rank * (self.qk_nope_head_dim + self.v_head_dim) # wkv_b
        num_ops += local_bsz * seqlen * self.n_local_heads * self.v_head_dim * self.hidden_size # wo
        num_ops += local_bsz * local_seqlen * self.n_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim) * seqlen # naive attention MACs

        hbm_reads = memory_footprint

        sp_comm_size = len(self.dist_info.attn_comm_groups["sp"])
        tp_comm_size = len(self.dist_info.attn_comm_groups["tp_attn"])

        network_data = 4 * intceil( (local_bsz * seqlen * self.n_local_heads * self.v_head_dim) / sp_comm_size) * (sp_comm_size -1) * dtype_to_byte(self.dtype) # allreduce_sp
        network_data += 4 * intceil( (local_bsz * seqlen * self.hidden_size) / tp_comm_size) * (tp_comm_size -1) * dtype_to_byte(self.dtype) # allreduce_tp

        expected = {
            "memory_footprint": memory_footprint,
            "num_ops": num_ops,
            "hbm_reads": hbm_reads,
            "network_data": network_data
        }
        return expected