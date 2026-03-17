import pytest
import json 

from src.node_level.layers.mla_naive import MLANaiveAttention
from src.node_level.common.tensor import Tensor
from src.node_level.common.stats import NodeStats
from src.node_level.common.utils import dtype_to_byte, intceil
from src.node_level.common.compute_graph import reset_compute_graph



@pytest.mark.parametrize(
    "bsz,seqlen,ctx_len,n_heads,qk_head_dim,v_head_dim,seq_parallel,dtype",
    [
        (1, 1, 128, 8, 192, 128, 1, "fp16"), # no batch, no seq parallelism
        (16, 1, 128, 8, 192, 128, 1, "fp16"), # batch case 
        (16, 1, 128, 8, 192, 128, 2, "fp16"), # batch and seq parallel
        (16, 1, 7777, 8, 192, 128, 4, "fp16"), # long context length, not equally divisible
    ]
)
def test_mla_absorb(bsz, seqlen, ctx_len, n_heads, qk_head_dim, v_head_dim, seq_parallel, dtype):
    reset_compute_graph()

    rank = 0
    stats = NodeStats()
    stats.new_iter(iter_id=0)

    op = MLANaiveAttention("mlanaive_0", rank, n_heads, qk_head_dim, v_head_dim, seq_parallel, dtype)
    q = Tensor("input", rank, [bsz, seqlen, n_heads, qk_head_dim])
    out_tensor = op.forward(q, ctx_len, stats=stats)

    expected = {
        "memory_footprint": bsz * intceil(ctx_len/seq_parallel) * n_heads * (qk_head_dim + v_head_dim) * dtype_to_byte(dtype), # kv_cache + pe_cache
        "num_ops": bsz * seqlen * intceil(ctx_len/seq_parallel) * n_heads * (qk_head_dim + v_head_dim), 
        "hbm_reads": bsz * intceil(ctx_len/seq_parallel) * n_heads * (qk_head_dim + v_head_dim) * dtype_to_byte(dtype) # kv_cache + pe_cache
    }

    assert out_tensor.dims == [bsz, seqlen, n_heads, v_head_dim]
    assert expected["memory_footprint"] == stats.get_stats("mlanaive_0")["memory_footprint"]
    assert expected["num_ops"] == stats.get_stats("mlanaive_0")["num_ops"]
    assert expected["hbm_reads"] == stats.get_stats("mlanaive_0")["hbm_reads"]

if __name__=="__main__":
    test_mla_absorb(16, 1, 128, 8, 192, 128, 1, "fp16")