
import logging

from utils import dtype_to_byte, intceil
from node_level.common.tensor import Tensor, View, Split, Concat, Slice, Transpose
from node_level.layers.linear import Linear
from node_level.layers.grouped_linear import GroupedLinear
from node_level.layers.allreduce import Allreduce
from node_level.layers.mla_absorb import MLAAbsorbAttention
from stats import NodeStats

class MLAAbsorbBlock:
    '''
    MLA Absorb Block.
    Args:
        uid: unique identifier for the layer
        hidden_size: hidden size of the model (DSv3: 7168)
        q_lora_rank: rank for query LoRA (DSv3: 1536)
        kv_lora_rank: rank for key/value LoRA (DSv3: 512)
        n_heads: total number of attention heads (DSv3: 128)
        qk_nope_head_dim: head dimension for query/key NOPE (DSv3: 128)
        qk_rope_head_dim: head dimension for query/key ROPE (DSv3: 64)
        v_head_dim: head dimension for value (DSv3: 128)
        dist_info: distributed information object
        next_layer: next layer in the model (not used here)
        dtype: data type of the weights (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype) -> None:
        super().__init__()
        logging.info("Creating MLA naive layer {}".format(uid))

        self.uid = uid
        self.dist_info = dist_info
        self.next_layer = next_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.rank = dist_info.rank

        self.n_local_heads = intceil(n_heads / dist_info.tp_attn)
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim

        self.ops = {}
        self.ops["wq_a"] = Linear(uid+"_wqa", self.rank, hidden_size, q_lora_rank, dtype)
        self.ops["wq_b"] = Linear(uid+"_wqb", self.rank, q_lora_rank, self.n_local_heads*self.qk_head_dim, dtype)
        self.ops["wkv_a"] = Linear(uid+"_wkva", self.rank, hidden_size, kv_lora_rank+qk_rope_head_dim, dtype)
        self.ops["wkv_b1"] = GroupedLinear(uid+"_wkvb1", self.rank, self.n_local_heads, qk_nope_head_dim, kv_lora_rank, dtype)
        self.ops["absorb_attn"] = MLAAbsorbAttention(uid+"_absorbattn", self.rank, self.n_local_heads, kv_lora_rank, qk_rope_head_dim, self.dist_info.sp, dtype)
        if dist_info.sp > 1:
            # Allreduce to aggregate attention output from sequence parallel devices
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", self.rank, self.n_local_heads * qk_rope_head_dim, dist_info.attn_comm_groups["sp"], dtype)
        
        self.ops["wkv_b2"] = GroupedLinear(uid+"_wkvb2", self.rank, self.n_local_heads, kv_lora_rank, v_head_dim, dtype)
        self.ops["wo"] = Linear(uid+"_wo", self.rank, self.n_local_heads*v_head_dim, hidden_size, dtype)
        if dist_info.tp_attn > 1:
            # Allreduce to aggregate output from tensor parallel devices
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", self.rank, hidden_size, dist_info.attn_comm_groups["tp_attn"], dtype)

        self._stats = NodeStats()

    '''
    Forward pass of MLA Absorb Block.
    Args:
        x: input tensor of shape (local_bsz, 1, hidden_size). seqlen is 1 for decode without speculative decoding.
        ctx_len: context length. determines the size of kv_cache and pe_cache
        stats: NodeStats object to record statistics
    '''
    def forward(self, x, ctx_len, stats):
        self._stats.new_iter(stats.iter)

        local_bsz, seqlen, _ = x.dims # input x is a minibatch based on a given dp factor
        assert seqlen == 1, "Absorb block should be used only for decode"

        kv = self.ops["wkv_a"].forward(x, stats=self._stats)
        #TODO: implement KV update

        q = self.ops["wq_a"].forward(x, stats=self._stats)
        q = self.ops["wq_b"].forward(q, stats=self._stats)

        q = View(q, [local_bsz, seqlen, self.n_local_heads, self.qk_head_dim]).forward(stats=self._stats)

        q_nope, q_pe = Split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1).forward(stats=self._stats)
        
        q_nope = View(q_nope, [local_bsz*seqlen, self.n_local_heads, self.qk_nope_head_dim]).forward(stats=self._stats)

        q_nope = Transpose(q_nope, [0, 1]).forward(stats=self._stats)

        q_nope = self.ops["wkv_b1"].forward(q_nope, stats=self._stats)

        q_nope = Transpose(q_nope, [0, 1]).forward(stats=self._stats)

        q_nope = View(q_nope, [local_bsz, seqlen, self.n_local_heads, self.kv_lora_rank]).forward(stats=self._stats)

        q = Concat([q_nope, q_pe], axis=-1).forward(stats=self._stats)

        attn_out = self.ops["absorb_attn"].forward(q, ctx_len, stats=self._stats)
        attn_out = Transpose(attn_out, [0, 2]).forward(stats=self._stats)

        if self.dist_info.sp > 1:
            attn_out = View(attn_out, [local_bsz, seqlen, self.n_local_heads*self.kv_lora_rank]).forward(stats=self._stats)
            attn_out = self.ops["allreduce_sp"].forward(attn_out, stats=self._stats)
            attn_out = View(attn_out, [local_bsz, seqlen, self.n_local_heads, self.kv_lora_rank]).forward(stats=self._stats)

        attn_out = View(attn_out, [self.n_local_heads, local_bsz*seqlen, self.kv_lora_rank]).forward(stats=self._stats)
        x = self.ops["wkv_b2"].forward(attn_out, stats=self._stats)
        x = Transpose(x, [0, 1]).forward(stats=self._stats)

        x = View(x, [local_bsz, seqlen, self.n_local_heads, self.v_head_dim]).forward(stats=self._stats)
        x = View(x, [local_bsz, seqlen, self.n_local_heads*self.v_head_dim]).forward(stats=self._stats)
        y = self.ops["wo"].forward(x, stats=self._stats)

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
        local_ctx_len = intceil(ctx_len / self.dist_info.sp)

        memory_footprint = self.hidden_size * self.q_lora_rank * dtype_to_byte(self.dtype) # wq_a
        memory_footprint += self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_to_byte(self.dtype) # wkv_a
        memory_footprint += self.q_lora_rank * self.n_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim) * dtype_to_byte(self.dtype) # wq_b
        memory_footprint += self.n_local_heads * self.qk_nope_head_dim * self.kv_lora_rank * dtype_to_byte(self.dtype) # wkv_b1
        memory_footprint += self.n_local_heads * self.kv_lora_rank * self.v_head_dim * dtype_to_byte(self.dtype) # wkv_b2
        memory_footprint += self.n_local_heads * self.v_head_dim * self.hidden_size * dtype_to_byte(self.dtype) # wo
        memory_footprint += local_bsz * local_ctx_len * (self.kv_lora_rank + self.qk_rope_head_dim) * dtype_to_byte(self.dtype) # kv_cache + pe_cache

        num_ops = local_bsz * seqlen * self.hidden_size * self.q_lora_rank # wq_a
        num_ops += local_bsz * seqlen * self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim) # wkv_a
        num_ops += local_bsz * seqlen * self.q_lora_rank * self.n_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim) # wq_b
        num_ops += local_bsz * seqlen * self.n_local_heads * self.qk_nope_head_dim * self.kv_lora_rank # wkv_b1
        num_ops += local_bsz * seqlen * self.n_local_heads * self.kv_lora_rank * self.v_head_dim # wkv_b2
        num_ops += local_bsz * seqlen * self.n_local_heads * self.v_head_dim * self.hidden_size # wo
        num_ops += local_bsz * seqlen * self.n_local_heads * (2 * self.kv_lora_rank + self.qk_rope_head_dim) * local_ctx_len # absorb attention MACs

        hbm_reads = memory_footprint

        sp_comm_size = len(self.dist_info.attn_comm_groups["sp"])
        tp_comm_size = len(self.dist_info.attn_comm_groups["tp_attn"])
        
        network_data = 4 * intceil( (local_bsz * seqlen * self.n_local_heads * self.kv_lora_rank) / sp_comm_size) * (sp_comm_size -1) * dtype_to_byte(self.dtype) # allreduce_sp
        network_data += 4 * intceil( (local_bsz * seqlen * self.hidden_size) / tp_comm_size) * (tp_comm_size -1) * dtype_to_byte(self.dtype) # allreduce_tp

        expected = {
            "memory_footprint": memory_footprint,
            "num_ops": num_ops,
            "hbm_reads": hbm_reads,
            "network_data": network_data
        }
        return expected