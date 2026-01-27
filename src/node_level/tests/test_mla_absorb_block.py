import pytest
import json 

from src.node_level.layers.mla_absorb_block import MLAAbsorbBlock
from src.node_level.common.tensor import Tensor
from src.node_level.common.stats import NodeStats
from src.node_level.common.compute_graph import reset_compute_graph
from src.node_level.common.config import SystemConfig


@pytest.mark.parametrize(
    "bsz,ctx_len,dp_attn,tp_attn,sp",
    [
        (4, 32, 1, 1, 1), # single node
        (4, 32, 2, 1, 1), # data parallel case
        (4, 32, 1, 2, 1), # tensor parallel case
        (4, 32, 1, 1, 2), # seq parallel case
        (64, 512, 4, 4, 4), # combined case
        (63, 512, 4, 4, 4), # uneven batch size case
        (63, 511, 4, 4, 4), # uneven batch size and context length case
    ]
)
def test_mla_absorb_block(bsz, ctx_len, dp_attn, tp_attn, sp):
    reset_compute_graph()

    stats = NodeStats()
    stats.new_iter(iter_id=0)

    seqlen = 1
    hidden_size = 7168
    q_lora_rank = 1536
    kv_lora_rank = 512
    n_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 512
    next_layer = None
    dtype = "fp16"

    num_nodes = dp_attn * tp_attn * sp
    decode_cfg = SystemConfig().from_args(
        num_nodes=num_nodes, 
        dp_attn=dp_attn, 
        tp_attn=tp_attn, 
        sp=sp, 
        ep=num_nodes # not relevant for MLA
    )

    assert n_heads % tp_attn == 0, "n_heads must be divisible by tp_attn"

    ops = {}
    for rank in range(num_nodes):
        dist_info = decode_cfg.get_dist_info(rank)

        ops[rank] = MLAAbsorbBlock(
            uid=f"{rank}:mlaabsorbblock_0", 
            hidden_size=hidden_size,  
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            n_heads=n_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            dist_info=dist_info,
            next_layer=next_layer,
            dtype=dtype
        )

        dist_info.batch_mapping(bsz)
        batch_ids = dist_info.get_local_batchids("attn")
        local_bsz = len(batch_ids)

        x = Tensor("input", rank, [local_bsz, seqlen, hidden_size])
        out_tensor = ops[rank].forward(x, ctx_len, stats=stats)

        assert out_tensor.dims == [local_bsz, seqlen, hidden_size]

        expected = ops[rank].calc_expected(local_bsz, seqlen, ctx_len)

        op_mem_foot, op_num_ops, op_hbm_reads, op_net_data = ops[rank]._stats.sumUp()
        assert expected["memory_footprint"] == op_mem_foot, f"Expected memory_footprint {expected['memory_footprint']}, got {op_mem_foot}"
        assert expected["num_ops"] == op_num_ops, f"Expected num_ops {expected['num_ops']}, got {op_num_ops}"
        assert expected["hbm_reads"] == op_hbm_reads, f"Expected hbm_reads {expected['hbm_reads']}, got {op_hbm_reads}"
        assert expected["network_data"] == op_net_data, f"Expected network_data {expected['network_data']}, got {op_net_data}"

if __name__=="__main__":
    test_mla_absorb_block(64, 512, 4, 4, 4)