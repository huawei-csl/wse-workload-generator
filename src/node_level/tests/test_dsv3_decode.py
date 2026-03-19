import json
import pytest 
import logging
from src.node_level.common.logger import init_logger

from src.node_level.generator import Generator
from src.node_level.common.config import SystemConfig
from src.node_level.models.model import build_model
from src.node_level.common.utils import dtype_to_byte, intceil

from src.node_level.common.workload import get_moe_gate_model, reset_moe_gate_model
from src.node_level.layers.moe import MoE
from src.core_level.common.graph import Graph

def run_decode(model_config, bsz, seqlen_q, prefill_len, decode_len, tp_attn=1, tp_ffn=1, dp_attn=1, dp_ffn=1, pp=1, ep=1, sp=1, moe_comm="alltoall", dtype="fp16"):
    reset_moe_gate_model()

    num_nodes = tp_attn * dp_attn * pp * sp

    generator = Generator()
    decode_cfg = SystemConfig().from_args(num_nodes, dp_attn, dp_ffn, tp_attn, tp_ffn, pp, sp, ep, n_redundant_shared_exp=1, expert_workload_model="uniform", moe_comm=moe_comm)

    models = []
    for rank in range(decode_cfg.num_nodes):
        model = build_model(model_config, decode_cfg.get_dist_info(rank), dtype, layer_ids="all", out_dir="/tmp/wse_workload_test")
        models.append(model)

    generator.decode(models, bsz, seqlen_q, prefill_len, decode_len, simplified_decode=True)

    # Check if we are able to build the graph correctly
    Graph(iter=f"decode0", num_nodes=num_nodes, dir=f"/tmp/wse_workload_test/graph")

    total_memory_footprint, total_num_ops, total_hbm_reads = 0, 0, 0
    num_activated_experts = 0
    for i in range(len(models)):
        mem_footprint, num_ops, hbm_reads, _ = models[i].stats.summarize()
        total_memory_footprint += mem_footprint
        total_num_ops += num_ops
        total_hbm_reads += hbm_reads

    layer_ids = [layer.ffn.uid for layer in models[0].layers if isinstance(layer.ffn, MoE)]
    for layer_id in layer_ids:
        num_activated_experts += len(get_moe_gate_model().get_activated_experts(layer_id))

    return total_memory_footprint, total_num_ops, total_hbm_reads, num_activated_experts

@pytest.mark.parametrize(
    "bsz,seqlen_q,dp_attn,tp_attn,sp,prefill_len,decode_len,moe_comm,dtype", 
    [
        (1, 1, 1, 1, 1, 1024, 100, "multicast", "fp16"), # single-batch, no parallelism
        (4, 1, 1, 1, 1, 1024, 100, "multicast", "fp16"), # multi-batch, no parallelism
        (8, 1, 2, 1, 1, 1024, 100, "multicast", "fp16"), # DP=2 in attention, EP=2 in FFN
        (8, 1, 1, 2, 1, 1024, 100, "multicast", "fp16"), # TP=2 in attention, EP=2 in FFN
        (8, 1, 1, 1, 2, 1024, 100, "multicast", "fp16"), # SP=2 in attention, EP=2 in FFN
        (8, 1, 2, 2, 2, 1024, 100, "multicast", "fp16"), # DP=2, TP=2, SP=2 in attention, EP=8 in FFN
        (8, 1, 2, 2, 2, 1024, 100, "multicast", "fp8"), # fp8
        (8, 1, 3, 2, 2, 1024, 100, "multicast", "fp8"), # uneven batch and expert split
        (128, 1, 3, 2, 2, 1024, 100, "multicast", "fp8"), # large batch size
        (8, 2, 3, 2, 2, 1024, 100, "multicast", "fp8"), # seqlen_q > 1 case, speculative decoding
        (1, 1, 1, 1, 1, 1024, 100, "alltoall", "fp16"), # single-batch, no parallelism
        (4, 1, 1, 1, 1, 1024, 100, "alltoall", "fp16"), # multi-batch, no parallelism
        (8, 1, 2, 1, 1, 1024, 100, "alltoall", "fp16"), # DP=2 in attention, EP=2 in FFN
        (8, 1, 1, 2, 1, 1024, 100, "alltoall", "fp16"), # TP=2 in attention, EP=2 in FFN
        (8, 1, 1, 1, 2, 1024, 100, "alltoall", "fp16"), # SP=2 in attention, EP=2 in FFN
        (8, 1, 2, 2, 2, 1024, 100, "alltoall", "fp16"), # DP=2, TP=2, SP=2 in attention, EP=8 in FFN
        (8, 1, 2, 2, 2, 1024, 100, "alltoall", "fp8"), # fp8
        (8, 1, 3, 2, 2, 1024, 100, "alltoall", "fp8"), # uneven batch and expert split
        (128, 1, 3, 2, 2, 1024, 100, "alltoall", "fp8"), # large batch size
        (8, 2, 3, 2, 2, 1024, 100, "alltoall", "fp8"), # seqlen_q > 1 case, speculative decoding
        (1, 1, 1, 1, 1, 1024, 100, "allgather", "fp16"), # single-batch, no parallelism
        (4, 1, 1, 1, 1, 1024, 100, "allgather", "fp16"), # multi-batch, no parallelism
        (8, 1, 2, 1, 1, 1024, 100, "allgather", "fp16"), # DP=2 in attention, EP=2 in FFN
        (8, 1, 1, 2, 1, 1024, 100, "allgather", "fp16"), # TP=2 in attention, EP=2 in FFN
        (8, 1, 1, 1, 2, 1024, 100, "allgather", "fp16"), # SP=2 in attention, EP=2 in FFN
        (8, 1, 2, 2, 2, 1024, 100, "allgather", "fp16"), # DP=2, TP=2, SP=2 in attention, EP=8 in FFN
        (8, 1, 2, 2, 2, 1024, 100, "allgather", "fp8"), # fp8
        (8, 1, 3, 2, 2, 1024, 100, "allgather", "fp8"), # uneven batch and expert split
        (128, 1, 3, 2, 2, 1024, 100, "allgather", "fp8"), # large batch size
        (8, 2, 3, 2, 2, 1024, 100, "allgather", "fp8"), # seqlen_q > 1 case, speculative decoding
    ]
)
def test_dsv3_decode(bsz, seqlen_q, dp_attn, tp_attn, sp, prefill_len, decode_len, moe_comm, dtype):
    '''
    This test counts the total number of operations for various EP values and expects it to be the same as single-node execution.
    '''
    with open("configs/deepseekv3.json", "r") as f:
        model_config = json.load(f) 

    init_logger(level=logging.ERROR, path='logs/test.log')

    num_nodes = dp_attn * tp_attn * sp
    tp_attn = tp_attn
    dp_attn = dp_attn
    sp = sp
    dp_ffn = 1
    tp_ffn = 1
    ep = num_nodes
    pp = 1

    _, total_num_ops, total_hbm_reads, num_activated_experts = run_decode(model_config, bsz, seqlen_q, prefill_len, decode_len, dp_attn=dp_attn, tp_attn=tp_attn, sp=sp, dp_ffn=dp_ffn, tp_ffn=tp_ffn, ep=ep, pp=pp, moe_comm=moe_comm, dtype=dtype)

    ctx_len = prefill_len + (decode_len -1)

    # Attention expected number of operations
    flops_wqa = (bsz/dp_attn) * seqlen_q * 11010048
    flops_wkva = (bsz/dp_attn) * seqlen_q * 4128768
    flops_wqb = (bsz/dp_attn) * seqlen_q * (37748736 // tp_attn)
    flops_wkvb1 = (bsz/dp_attn) * seqlen_q * (8388608 // tp_attn)
    flops_wkvb2 = (bsz/dp_attn) * seqlen_q * (8388608 // tp_attn)
    flops_wo = (bsz/dp_attn) * seqlen_q * (117440512 // tp_attn)
    flops_absorb_attn = (bsz/dp_attn) * seqlen_q * intceil(ctx_len / sp) * (128 // tp_attn) * 1088
    expected_flops_attn = flops_wqa + flops_wkva + flops_wqb + flops_wkvb1 + flops_wkvb2 + flops_wo + flops_absorb_attn
    expected_flops_attn = num_nodes * expected_flops_attn

    # MoE expected number of operations
    n_routed_experts = 256
    moe_weight_size = 3 * 7168 * 2048
    expected_flops_moe = bsz * seqlen_q * 9 * moe_weight_size # calculations with routed experts
    expected_flops_moe += num_nodes * (bsz / dp_attn) * seqlen_q * 7168 * n_routed_experts # gating network 

    dense_ffn_weight_size = 3 * 7168 * 18432
    expected_flops_dense = bsz * seqlen_q * dense_ffn_weight_size # dense FFN

    expected_flops_lm_head = bsz * seqlen_q * 7168 * 129280 # LM head

     # 3 first dense layers + 58 MoE layers + 1 LM head
    expected_flops = 3 * (expected_flops_attn + expected_flops_dense)
    expected_flops += 58 * (expected_flops_attn + expected_flops_moe)
    expected_flops += expected_flops_lm_head
    expected_flops = round(expected_flops)

    # Attention HBM reads expected values
    mem_wqa = 11010048 * dtype_to_byte(dtype)
    mem_wkva = 4128768 * dtype_to_byte(dtype)
    mem_wqb = (37748736 // tp_attn) * dtype_to_byte(dtype)
    mem_wkvb1 = (8388608 // tp_attn) * dtype_to_byte(dtype)
    mem_wkvb2 = (8388608 // tp_attn) * dtype_to_byte(dtype)
    mem_wo = (117440512 // tp_attn) * dtype_to_byte(dtype)
    mem_kv_cache = (bsz / dp_attn) * intceil(ctx_len / sp) * 576 * dtype_to_byte(dtype)
    expected_hbm_reads_attn = mem_wqa + mem_wkva + mem_wqb + mem_wkvb1 + mem_wkvb2 + mem_wo + mem_kv_cache
    expected_hbm_reads_attn = num_nodes * expected_hbm_reads_attn

    # MoE HBM reads expected values
    avg_num_activated_experts = num_activated_experts / 58
    expected_hbm_reads_moe = avg_num_activated_experts * moe_weight_size * dtype_to_byte(dtype) 
    expected_hbm_reads_moe += moe_weight_size * dtype_to_byte(dtype) # weights for shared expert
    expected_hbm_reads_moe += num_nodes * 7168 * n_routed_experts * dtype_to_byte(dtype) # gating network 

    # dense FFN weights
    expected_hbm_reads_dense = num_nodes * (dense_ffn_weight_size // (tp_attn * sp)) * dtype_to_byte(dtype) # dense FFN weights

    # LM head weights
    expected_hbm_reads_lmhead = num_nodes * (7168 * 129280 // (tp_attn * sp)) * dtype_to_byte(dtype)

    # 3 first dense layers + 58 MoE layers + 1 LM head
    expected_hbm_reads = 3 * (expected_hbm_reads_attn + expected_hbm_reads_dense)
    expected_hbm_reads += 58 * (expected_hbm_reads_attn + expected_hbm_reads_moe)
    expected_hbm_reads += expected_hbm_reads_lmhead
    expected_hbm_reads = round(expected_hbm_reads)

    print("Total FLOPs expected: ", expected_flops)
    print("Total FLOPs actual:   ", total_num_ops)

    print("Total HBM reads expected: ", expected_hbm_reads)
    print("Total HBM reads actual:   ", total_hbm_reads)

    assert expected_flops == total_num_ops, f"Num ops mismatch: {expected_flops} vs {total_num_ops}"
    assert expected_hbm_reads == total_hbm_reads, f"HBM reads mismatch: {expected_hbm_reads} vs {total_hbm_reads}"

if __name__ == "__main__":
    test_dsv3_decode(8, 2, 3, 2, 2, 1024, 100, "alltoall", "fp8")