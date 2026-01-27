import pytest 

from src.core_level.common.wafer import Wafer
from src.core_level.common.tensor import reset_tensor_registry
from src.core_level.common.graph import Graph
from src.core_level.common.tensor import Tensor
from src.core_level.layers.attention import MLALayer

@pytest.mark.parametrize(
    "bsz, seqlen_q, seqlen_kv, num_heads, kv_lora_rank, qk_rope_head_dim, q_tile_size, kv_tile_size",
    [
        (4, 1, 32, 8, 64, 16, [4, 1, 8, 64], [4, 32, 64]), # no tiling
        (4, 1, 32, 8, 64, 16, [4, 1, 2, 64], [4, 32, 64]), # tiling on attention heads
        (4, 1, 32, 8, 64, 16, [4, 1, 8, 64], [4, 4, 64]), # tiling on KV sequence length
        (16, 1, 2048, 128, 512, 64, [1, 1, 8, 512], [1, 128, 512]), # DSv3 params, with attention and KV tiling
    ]
)

def test_attention(bsz, seqlen_q, seqlen_kv, num_heads, kv_lora_rank, qk_rope_head_dim, q_tile_size, kv_tile_size):
    reset_tensor_registry()

    node_grid = (1, 1)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:mla_0"
        ops[node_id][op_id] = {
            "type": "MLAAbsorbAttention",
            "inputs": [f"{node_id}:mla_0_q"],
            "outputs": [f"{node_id}:output_tensor"]
        }

    q_dims = bsz, seqlen_q, num_heads, kv_lora_rank
    kv_dims = bsz, seqlen_kv, kv_lora_rank
    pe_dims = bsz, seqlen_kv, qk_rope_head_dim

    q_tensor = Tensor(
        uid=f"{node_id}:mla_0_q",
        dims=[bsz, seqlen_q, num_heads, kv_lora_rank+qk_rope_head_dim],
        prec="fp16",
    )
    
    kv_tensor = Tensor(
        uid=f"{node_id}:mla_0_kv",
        dims=[bsz, seqlen_kv, kv_lora_rank+qk_rope_head_dim],
        prec="fp16",
    )

    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    for node_id in range(wafer.num_nodes):
        layer = MLALayer(f"{node_id}:mla_0", node_id, graph, q_dims, kv_dims, pe_dims, q_tile_size, kv_tile_size, wafer, prec="fp16")
    
    expected = layer.calc_expected()

    B, S_q, H, D = layer.q_dims
    B, S_kv, D = layer.kv_dims
    B, S_kv, D_pe = layer.pe_dims
    Tb, Ts_q, Th, Td = layer.q_tile_size
    Tb, Ts_kv, Td = layer.kv_tile_size

    q_size = expected['q_size']
    kv_reads = expected['kv_size']
    output_size = expected["output_size"]

    if S_kv == Ts_kv:
        total_reads = kv_reads * (H // Th)
        total_reads += q_size
        total_writes = output_size
    else:
        total_reads = kv_reads * (H // Th)
        total_reads += q_size * (S_kv // Ts_kv) # needs to reread q for each kv tile 
        total_reads += output_size * (S_kv // Ts_kv) # needs to read partial output tiles for accumulation
        
        total_writes = output_size * (S_kv // Ts_kv + 1) # needs to write partial output tiles plus final output

    assert total_reads == layer.stats.get_reads(), f"Expected {total_reads} reads, but got {layer.stats.get_reads()}"
    assert total_writes == layer.stats.get_writes(), f"Expected {total_writes} writes, but got {layer.stats.get_writes()}"
    assert expected["flops"] == layer.stats.get_total_cube(), f"Expected {expected['flops']} flops, but got {layer.stats.get_total_cube()}"

if __name__=="__main__":
    bsz = 4
    seqlen_q = 1
    seqlen_kv = 32
    num_heads = 16
    kv_lora_rank = 64
    qk_rope_head_dim = 16
    q_tile_size = [1, 1, 4, 64]
    kv_tile_size = [1, 4, 64]

    test_attention(bsz, seqlen_q, seqlen_kv, num_heads, kv_lora_rank, qk_rope_head_dim, q_tile_size, kv_tile_size)
