import logging

from node_level.common.tensor import Tensor
from utils import dtype_to_byte, intceil
from node_level.common.compute_graph import get_compute_graph

class MLAAbsorbAttention:
    '''
    MLA Absorb Attention Operator
    Inputs:
        uid: unique identifier for the layer
        rank: rank of the node where this layer is located
        n_local_heads: number of local attention heads
        kv_lora_rank: kv_lora_rank
        qk_rope_head_dim: qk_rope_head_dim
        seq_parallel: sequence parallelism degree
        dtype: data type of the weights (e.g., "fp16", "fp8")    
    '''
    def __init__(self, uid, rank, n_local_heads, kv_lora_rank, qk_rope_head_dim, seq_parallel, dtype) -> None:
        super().__init__()

        self.uid = uid 
        self.rank = rank
        self.n_local_heads = n_local_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.seq_parallel = seq_parallel
        self.dtype = dtype

    def forward(self, x, ctx_len=None, stats=None):
        '''
        Forward pass of MLA Absorb Attention.
        Args:
            x: input tensor of shape (bsz, seqlen, n_local_heads, kv_lora_rank + qk_rope_head_dim). 
                seqlen is 1 for decode without speculative decoding.
                last dimension is concatenation of kv_lora and qk_rope
            ctx_len: context length. determines the size of kv_cache and pe_cache
            stats: NodeStats object to record statistics
        Returns:
            out: output tensor of shape (bsz, seqlen, n_local_heads, kv_lora_rank)
        '''
        assert len(x.dims) == 4, "Input tensor must be 4D (bsz, seqlen, n_local_heads, head_dim)"

        bsz, seqlen, n_local_heads, D = x.dims
        assert n_local_heads == self.n_local_heads, "Input n_local_heads {} does not match n_local_heads {}".format(n_local_heads, self.n_local_heads)
        assert D == self.kv_lora_rank + self.qk_rope_head_dim, "Input head dim {} does not match kv_lora_rank+qk_rope_head_dim {}".format(D, self.kv_lora_rank+self.qk_rope_head_dim)

        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)
        dims = self.get_dims(bsz, seqlen, ctx_len)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        out = Tensor(f"{self.uid}_out", self.rank, (bsz, seqlen, n_local_heads, self.kv_lora_rank))
        stats.append(self.uid, "MLAAbsorbAttention", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, bsz, ctx_len):
        ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
        memory_footprint = bsz * ctx_len_per_device * self.kv_lora_rank * dtype_to_byte(self.dtype) # kv_cache
        memory_footprint += bsz * ctx_len_per_device * self.qk_rope_head_dim * dtype_to_byte(self.dtype) # pe_cache
        return memory_footprint  # KV-cache only, in bytes
    
    def get_dims(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.seq_parallel)
            input_dims = [bsz, seqlen, self.n_local_heads, self.kv_lora_rank]
            K_dims = [bsz, seqlen_per_device, self.kv_lora_rank]
            V_dims = [bsz, seqlen_per_device, self.qk_rope_head_dim]
            out_dims = [bsz, seqlen, self.n_local_heads, self.kv_lora_rank]
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            input_dims = [bsz, 1, self.n_local_heads, self.kv_lora_rank]
            K_dims = [bsz, ctx_len_per_device, self.kv_lora_rank]
            V_dims = [bsz, ctx_len_per_device, self.qk_rope_head_dim]
            out_dims = [bsz, 1, self.n_local_heads, self.kv_lora_rank]
        return "Q: " + str(input_dims) + ", KV: " + str(K_dims) + ", PE: " + str(V_dims) + " -> " + str(out_dims)
    
    def num_ops(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.seq_parallel)
            n_ops = bsz * seqlen_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bshc,btc→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.qk_rope_head_dim * seqlen # einsum(bshr,btr→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bsht,btc→bshc)
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            n_ops = bsz * ctx_len_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bshc,btc→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.qk_rope_head_dim * seqlen # einsum(bshr,btr→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bsht,btc→bshc)
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
        rw = bsz * ctx_len_per_device * self.kv_lora_rank * dtype_to_byte(self.dtype) # kv_cache
        rw += bsz * ctx_len_per_device * self.qk_rope_head_dim * dtype_to_byte(self.dtype) # pe_cache
        return rw # KV-cache only, in bytes

    def network_data(self, bsz=None):
        return 0
