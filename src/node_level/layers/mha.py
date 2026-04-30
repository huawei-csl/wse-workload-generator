
import logging

from src.node_level.common.tensor import Tensor
from src.node_level.common.utils import dtype_to_byte, intceil
from src.node_level.common.compute_graph import get_compute_graph

class SelfAttention:
    def __init__(self, uid, rank, num_attention_heads, num_key_value_heads, head_dim, seq_parallel, dtype) -> None:
        super().__init__()

        logging.debug("SelfAttention layer {} with KV-cache dims: bsz x ctx_len x {} x {}".format(uid, num_key_value_heads, head_dim))

        self.uid = uid
        self.rank = rank 
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.seq_parallel = seq_parallel

    def forward(self, x, ctx_len=None, stats=None):
        assert len(x.dims) == 4, "Input tensor must be 4D (bsz, seqlen, n_local_heads, head_dim)"

        bsz, seqlen, n_local_heads, head_dim = x.dims
        assert n_local_heads == self.num_attention_heads, "Input n_local_heads {} does not match n_local_heads {}".format(n_local_heads, self.num_attention_heads)
        assert head_dim == self.head_dim, "Input head dim {} does not match head_dim {}".format(head_dim, self.head_dim)

        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)
        dims = self.get_dims(bsz, seqlen, ctx_len)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        out = Tensor(f"{self.uid}_out", self.rank, (bsz, seqlen, self.num_attention_heads, self.head_dim))
        stats.append(self.uid, "SelfAttention", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)
        return out 
    
    def memory_footprint(self, bsz, ctx_len):
        memory_footprint = 2 * bsz * intceil(ctx_len/self.seq_parallel) * self.num_key_value_heads * self.head_dim * dtype_to_byte(self.dtype) # KV-cache
        return memory_footprint  # KV-cache only, in bytes

    def get_dims(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.seq_parallel)
            input_dims = [bsz, seqlen, self.n_local_heads, self.qk_head_dim]
            K_dims = [bsz, seqlen_per_device, self.n_local_heads, self.qk_head_dim]
            V_dims = [bsz, seqlen_per_device, self.n_local_heads, self.v_head_dim]
            out_dims = [bsz, seqlen, self.n_local_heads, self.v_head_dim]
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            input_dims = [bsz, 1, self.n_local_heads, self.qk_head_dim]
            K_dims = [bsz, ctx_len_per_device, self.n_local_heads, self.qk_head_dim]
            V_dims = [bsz, ctx_len_per_device, self.n_local_heads, self.v_head_dim]
            out_dims = [bsz, 1, self.n_local_heads, self.v_head_dim]
        return ",".join(input_dims) + " x " + ",".join(K_dims) + " x " + ",".join(V_dims) + " -> " + ",".join(out_dims)
    
    def get_dims(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.seq_parallel)
            input_dims = [bsz, seqlen, self.num_attention_heads, self.head_dim]
            K_dims = [bsz, seqlen_per_device, self.num_key_value_heads, self.head_dim]
            V_dims = [bsz, seqlen_per_device, self.num_key_value_heads, self.head_dim]
            out_dims = [bsz, seqlen, self.num_attention_heads, self.head_dim]
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            input_dims = [bsz, 1, self.num_attention_heads, self.head_dim]
            K_dims = [bsz, ctx_len_per_device, self.num_key_value_heads, self.head_dim]
            V_dims = [bsz, ctx_len_per_device, self.num_key_value_heads, self.head_dim]
            out_dims = [bsz, 1, self.num_attention_heads, self.head_dim]
        return "Q: " + str(input_dims) + ", K: " + str(K_dims) + ", V: " + str(V_dims) + " -> " + str(out_dims)
    
    def num_ops(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            n_ops = bsz * intceil(seqlen/self.seq_parallel) * self.num_attention_heads * self.head_dim * seqlen # QKT
            n_ops += bsz * intceil(seqlen/self.seq_parallel) * self.num_attention_heads * self.head_dim * seqlen # SV
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            logging.debug("{} bsz: {}, ctx_len: {}, num_attention_heads: {}, head_dim: {}, seqlen: {}".format(self.uid, bsz, ctx_len_per_device, self.num_attention_heads, self.head_dim, seqlen))
            n_ops = bsz * ctx_len_per_device * self.num_attention_heads * self.head_dim * seqlen # QKT
            n_ops += bsz * ctx_len_per_device * self.num_attention_heads * self.head_dim * seqlen # SV
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
        logging.debug("{} bsz: {}, ctx_len: {}, num_attention_heads: {}, head_dim: {}".format(self.uid, bsz, ctx_len_per_device, self.num_attention_heads, self.head_dim))
        rw = 2 * bsz * ctx_len_per_device * self.num_key_value_heads * self.head_dim * dtype_to_byte(self.dtype) # KV-cache
        return rw # KV-cache only, in bytes

    def network_data(self, bsz=None):
        return 0
