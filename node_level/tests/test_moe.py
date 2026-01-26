import pytest
import json 
import numpy as np 

from node_level.layers.moe import MoE
from node_level.common.tensor import Tensor, Slice
from node_level.common.compute_graph import reset_compute_graph

from stats import NodeStats
from config import SystemConfig
from utils import dtype_to_byte
from workload import get_moe_gate_model, reset_moe_gate_model


@pytest.mark.parametrize(
    "bsz,ep,dp_attn,n_redundant_shared_exp,dtype", 
    [
        (1, 8, 1, 1, "fp16"), # single-batch test case
        (4, 8, 1, 1, "fp16"), # multi-batch test case
        (128, 8, 1, 1, "fp16"), # large-batch test case
        (128, 8, 2, 1, "fp16"), # dp_attn > 1
        (128, 8, 8, 1, "fp16"), # dp_attn == ep
        (128, 8, 2, 2, "fp16"), # with redundant shared experts
        (128, 8, 2, 8, "fp16"), # each node has a redundant shared expert
        (128, 56, 14, 4, "fp16"), # unbalanced num local experts
        (128, 56, 14, 4, "fp8"), # fp8
    ], 
)
def test_moe(bsz, ep, dp_attn, n_redundant_shared_exp, dtype):
    reset_moe_gate_model()
    reset_compute_graph()

    seqlen = 1

    hidden_size = 7168
    moe_intermediate_size = 2048
    num_experts_per_tok = 8
    n_experts = 256
    n_shared_experts = 1

    num_nodes = ep
    tp_attn = num_nodes // dp_attn
    for rank in range(num_nodes):
        reset_compute_graph()

        stats = NodeStats()
        stats.new_iter(iter_id=0)

        dist_info = SystemConfig().from_args(
                num_nodes=num_nodes,
                dp_attn=dp_attn,
                tp_attn=tp_attn,
                ep=ep,
                moe_comm="multicast",
                expert_workload_model="uniform",
                n_redundant_shared_exp=n_redundant_shared_exp,
        ).get_dist_info(rank)
        dist_info.batch_mapping(bsz)

        moe_gate_model = get_moe_gate_model(num_experts_per_tok, n_experts, ["moe_0"], dist_info.expert_workload_model)
        moe_gate_model.new_iter(iter_id=0, bsz=bsz, seqlen=seqlen)

        moe_layer = MoE(
            uid=f"moe_0",
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            n_experts=n_experts,
            n_shared_experts=n_shared_experts,
            dist_info=dist_info,
            dtype=dtype
        )

        batch_ids = dist_info.get_local_batchids("attn")
        local_bsz = len(batch_ids)

        x = Tensor("input", dist_info.rank, [bsz, seqlen, hidden_size])
        x_local = Slice(x, [batch_id for batch_id in batch_ids], axis=0).forward(stats=stats)

        y = moe_layer.forward(x_local, stats=stats)

        assert y.dims == [local_bsz, seqlen, hidden_size], f"Output dims {y.dims} do not match expected dims {[local_bsz, seqlen, hidden_size]}"

        op_mem_foot, op_num_ops, op_hbm_reads, op_net_data = stats.sumUp()
        print(op_mem_foot, op_num_ops, op_hbm_reads, op_net_data)

        expert_routing = moe_gate_model.get_expert_routings(layer_id="moe_0")

        local_experts = []
        for expert_id, rank_ep in dist_info.get_expert_mapping(n_experts).items():
            if rank == rank_ep:
                local_experts.append(expert_id)

        mem_footprint_per_expert = 3 * hidden_size * moe_intermediate_size * dtype_to_byte(dtype) # weights only
        
        expected_footprint = hidden_size * n_experts * dtype_to_byte(dtype) # gate weights
        expected_footprint += len(local_experts) * mem_footprint_per_expert # local routed experts
        # if this node holds a shared expert
        if rank in dist_info.shared_expert_ranks:
            expected_footprint += mem_footprint_per_expert

        expected_num_ops = local_bsz * seqlen * hidden_size * n_experts # gating
        expected_hbm_reads = hidden_size * n_experts * dtype_to_byte(dtype) # gating
        for expert_id in local_experts:
            num_tokens = len(np.where(expert_routing==expert_id)[1])
            expected_num_ops += 3 * num_tokens * hidden_size * moe_intermediate_size
            if num_tokens > 0:
                expected_hbm_reads += 3 * hidden_size * moe_intermediate_size * dtype_to_byte(dtype) # weights

        batch_ids_for_shared = [batch_id for batch_id, mapped_shared in dist_info.batch_to_shared_exp.items() if dist_info.rank == mapped_shared]
        if rank in dist_info.shared_expert_ranks:
            expected_num_ops += 3 * len(batch_ids_for_shared) * hidden_size * (moe_intermediate_size * n_shared_experts)
            expected_hbm_reads += 3 * hidden_size * (moe_intermediate_size * n_shared_experts) * dtype_to_byte(dtype) # weights

        dispatch_traffic, combine_traffic = moe_layer.routings_summary()

        expected_network_data = 0
        src_id = rank
        for dst_id in range(len(dispatch_traffic)):
            expected_network_data += len(dispatch_traffic[src_id][dst_id]) * seqlen * hidden_size * dtype_to_byte(dtype)

        for dst_id in range(len(combine_traffic)):
            expected_network_data += len(combine_traffic[src_id][dst_id]) * seqlen * hidden_size * dtype_to_byte(dtype)

        dp_attn_cluster_size = len(dist_info.dp_attn_cluster)
        if dist_info.is_dp_master():
            expected_network_data += local_bsz * seqlen * hidden_size * dtype_to_byte(dtype) * (dp_attn_cluster_size-1)

        assert expected_footprint == moe_layer.memory_footprint(), f"Expected memory_footprint {expected_footprint}, got {moe_layer.memory_footprint()}"
        assert expected_num_ops == op_num_ops, f"Expected num_ops {expected_num_ops}, got {op_num_ops}"
        assert expected_hbm_reads == op_hbm_reads, f"Expected hbm_reads {expected_hbm_reads}, got {op_hbm_reads}"
        assert expected_network_data == op_net_data, f"Expected network data {expected_network_data}, got {op_net_data}"


if __name__=="__main__":
    test_moe(
        bsz=1,
        ep=8,
        dp_attn=1,
        n_redundant_shared_exp=1,
        dtype="fp16"
    )