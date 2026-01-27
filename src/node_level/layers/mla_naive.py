
import logging

from src.node_level.common.utils import intceil, dtype_to_byte

class MLANaiveAttention:
    def __init__(self, uid, n_local_heads, qk_head_dim, v_head_dim, dist_info, dtype) -> None:
        super().__init__()

        self.uid = uid 
        self.n_local_heads = n_local_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.dtype = dtype
        self.dist_info = dist_info

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        raise NotImplementedError("Not yet implemented, ask for support")
        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)
        dims = self.get_dims(bsz, seqlen, ctx_len)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        stats.append(self.uid, "MLANaiveAttention", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)

    def memory_footprint(self, bsz, ctx_len):
        memory_footprint = bsz * intceil(ctx_len/self.dist_info.sp) * self.n_local_heads * self.qk_head_dim * dtype_to_byte(self.dtype) # k_cache
        memory_footprint += bsz * intceil(ctx_len/self.dist_info.sp) * self.n_local_heads * self.v_head_dim * dtype_to_byte(self.dtype) # v_cache
        return memory_footprint  # KV-cache only, in bytes

    def get_dims(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.dist_info.sp)
            input_dims = [bsz, seqlen, self.n_local_heads, self.qk_head_dim]
            K_dims = [bsz, seqlen_per_device, self.n_local_heads, self.qk_head_dim]
            V_dims = [bsz, seqlen_per_device, self.n_local_heads, self.v_head_dim]
            out_dims = [bsz, seqlen, self.n_local_heads, self.v_head_dim]
        else:
            ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
            input_dims = [bsz, 1, self.n_local_heads, self.qk_head_dim]
            K_dims = [bsz, ctx_len_per_device, self.n_local_heads, self.qk_head_dim]
            V_dims = [bsz, ctx_len_per_device, self.n_local_heads, self.v_head_dim]
            out_dims = [bsz, 1, self.n_local_heads, self.v_head_dim]
        return "Q: " + str(input_dims) + ", K: " + str(K_dims) + ", V: " + str(V_dims) + " -> " + str(out_dims)
    
    def num_ops(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.dist_info.sp)
            n_ops = bsz * seqlen_per_device * self.n_local_heads * self.qk_head_dim * seqlen # einsum(bshd,bthd→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.v_head_dim * seqlen # einsum(bsht,bthv→bshv)
        else:
            ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
            n_ops = bsz * ctx_len_per_device * self.n_local_heads * self.qk_head_dim * seqlen # einsum(bshd,bthd→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.v_head_dim * seqlen # einsum(bsht,bthv→bshv)
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
        rw = bsz * ctx_len_per_device * self.n_local_heads * self.qk_head_dim * dtype_to_byte(self.dtype) # k_cache
        rw += bsz * ctx_len_per_device * self.n_local_heads * self.v_head_dim * dtype_to_byte(self.dtype) # v_cache
        return rw # KV-cache only, in bytes

    def network_data(self, bsz=None):
        return 0
    