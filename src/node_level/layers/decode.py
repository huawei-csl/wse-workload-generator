
import logging

from src.node_level.common.tensor import Tensor
from src.node_level.layers.ffn import FFN
from src.node_level.layers.moe import MoE

from src.node_level.layers.mla_naive_block import MLANaiveBlock
from src.node_level.layers.mla_absorb_block import MLAAbsorbBlock

from src.node_level.common.utils import intceil

class LlamaDecodeLayer:
    def __init__(self, layer_id, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, dist_info, dtype) -> None:
        super().__init__()
        raise NotImplementedError
    
        logging.info("Creating Decode layer {}".format(layer_id))

        self.dist_info = dist_info

        self.attention = GQABlock(layer_id+"_attn", hidden_size, num_attention_heads, num_key_value_heads, dist_info, dtype)
        self.ffn = FFN(layer_id+"_ffn", hidden_size, intermediate_size, dist_info, dtype)

    def forward(self, queries, ctx_len, stats):
        bsz, seqlen, _ = queries.dims
        self.attention.forward(bsz, seqlen, ctx_len, stats=stats)

        seqlen_per_device_ffn = intceil(seqlen/self.dist_info.sp) # This is only effective in prefill, seqlen=1 in decode anyway
        self.ffn.forward(bsz*seqlen_per_device_ffn, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        bsz_per_device_attn = intceil(bsz/self.dist_info.dp_attn)
        mem_size = self.attention.memory_footprint(bsz_per_device_attn, ctx_len)

        bsz_per_device_ffn = intceil(bsz/self.dist_info.dp_ffn)
        mem_size += self.ffn.memory_footprint(bsz_per_device_ffn)

        return mem_size # in bytes

class MLABlock:
    def __init__(self, uid, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype) -> None:
        super().__init__()
        logging.info("Creating MLA block {}".format(uid))

        self.uid = uid
        self.dist_info = dist_info

        self.MLA_naive = MLANaiveBlock(uid+"_naive", hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype)
        self.MLA_absorb = MLAAbsorbBlock(uid+"_absorb", hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype)

    def forward(self, x, ctx_len, stats):
        is_prefill = ctx_len == 0

        if is_prefill:
            return self.MLA_naive.forward(x, ctx_len, stats)
        else:
            return self.MLA_absorb.forward(x, ctx_len, stats)

    def set_next_layer(self, next_layer):
        self.MLA_naive.next_layer = next_layer
        self.MLA_absorb.next_layer = next_layer

    def memory_footprint(self, bsz, ctx_len):
        mem_size = self.MLA_absorb.memory_footprint(bsz, ctx_len)
        return mem_size # in bytes


class DSv3DecodeLayer:
    def __init__(self, layer_id, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, intermediate_size, moe_intermediate_size, num_experts_per_tok, n_experts, n_shared_experts, dist_info, dtype, is_moe=False) -> None:
        super().__init__()
        logging.info("Creating Decode layer {}".format(layer_id))

        self.layer_id = layer_id
        self.dist_info = dist_info

        self.attention = MLABlock(layer_id+"_attn", hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer=None, dtype=dtype)

        if is_moe:
            self.ffn = MoE(layer_id+"_moe", hidden_size, moe_intermediate_size, num_experts_per_tok, n_experts, n_shared_experts, dist_info, dtype)
        else:
            self.ffn = FFN(layer_id+"_dense", hidden_size, intermediate_size, dist_info, dtype, is_dense_layer=True)

        self.attention.set_next_layer(self.ffn)

    def forward(self, x, ctx_len, stats):
        bsz, seqlen, _ = x.dims

        is_prefill = ctx_len == 0
        if not is_prefill:
            assert seqlen == 1

        x = self.attention.forward(x, ctx_len, stats=stats)

        self.ffn.forward(x, stats=stats)

        return Tensor(self.layer_id+"_out", self.dist_info.rank, x.dims)

    def memory_footprint(self, bsz, ctx_len):
        batch_ids = self.dist_info.get_local_batchids("attn")
        bsz_per_device_attn = len(batch_ids)
        mem_size = self.attention.memory_footprint(bsz_per_device_attn, ctx_len)

        batch_ids = self.dist_info.get_local_batchids("ffn")
        bsz_per_device_ffn = len(batch_ids)
        mem_size += self.ffn.memory_footprint(bsz_per_device_ffn)

        return mem_size # in bytes
