import pytest 

from src.node_level.common.stats import NodeStats
from src.node_level.common.utils import dtype_to_byte, intceil
from src.node_level.common.config import SystemConfig

from src.node_level.layers.ffn import FFN
from src.node_level.common.tensor import Tensor
from src.node_level.common.compute_graph import reset_compute_graph

@pytest.mark.parametrize(
    "bsz,hidden_size,intermediate_size,tp_ffn,ep,dtype",
    [
        (4, 1024, 512, 1, 1, "fp16"), # single node
        (4, 1024, 512, 2, 1, "fp16"), # tp_ffn = 2
        (8, 1024, 512, 3, 1, "fp16"), # tp_ffn = 3 uneven case
        (16, 2048, 1024, 4, 1, "fp8"), # tp_ffn = 4 with fp8
    ]
)
def test_ffn(bsz, hidden_size, intermediate_size, tp_ffn, ep, dtype):
    reset_compute_graph()

    rank = 0
    stats = NodeStats()
    stats.new_iter(iter_id=0)

    seqlen = 1

    num_nodes = tp_ffn * ep
    decode_cfg = SystemConfig().from_args(
        num_nodes=num_nodes, 
        dp_attn=num_nodes, # not relevant for FFN
        dp_ffn=1,
        tp_ffn=tp_ffn,
        ep=ep
    )

    for rank in range(num_nodes):
        dist_info = decode_cfg.get_dist_info(rank)
        dist_info.batch_mapping(bsz)
        batch_ids = dist_info.get_local_batchids("attn")
        local_bsz = len(batch_ids)

        op = FFN(f"{rank}ffn_0", hidden_size, intermediate_size, dist_info, dtype, is_dense_layer=False)

        x = Tensor("input", rank, [local_bsz, seqlen, hidden_size])

        y = op.forward(x, stats=stats)

        assert y.dims == [local_bsz, seqlen, hidden_size]

        par_factor = tp_ffn
        comm_group = dist_info.ffn_comm_groups["tp_ffn"]
        local_inter_size = intceil(intermediate_size / par_factor)
        ffn_comm_size = len(comm_group)

        expected = {
            "memory_footprint": 3 * hidden_size * local_inter_size * dtype_to_byte(dtype), # weights only
            "num_ops": 3 * local_bsz * seqlen * hidden_size * local_inter_size, # in terms of MACs
            "hbm_reads": 3 * hidden_size * local_inter_size * dtype_to_byte(dtype), # weights only
            "network_data": 4 * intceil( (local_bsz * seqlen * hidden_size) / ffn_comm_size) * (ffn_comm_size -1) * dtype_to_byte(dtype) # allreduce for tp_ffn
        }

        op_mem_foot, op_num_ops, op_hbm_reads, op_net_data = op._stats.sumUp()
        assert expected["memory_footprint"] == op_mem_foot, f"Expected memory_footprint {expected['memory_footprint']}, got {op_mem_foot}"
        assert expected["num_ops"] == op_num_ops, f"Expected num_ops {expected['num_ops']}, got {op_num_ops}"
        assert expected["hbm_reads"] == op_hbm_reads, f"Expected hbm_reads {expected['hbm_reads']}, got {op_hbm_reads}"
        assert expected["network_data"] == op_net_data, f"Expected network_data {expected['network_data']}, got {op_net_data}"

@pytest.mark.parametrize(
    "bsz,hidden_size,intermediate_size,dp_attn,tp_attn,sp,dtype",
    [
        (4, 1024, 512, 1, 1, 1, "fp16"), # single node
        (32, 1024, 512, 4, 1, 1, "fp16"), # dp-only
        (32, 1024, 512, 4, 2, 1, "fp16"), # dp + tp
        (32, 1024, 512, 4, 2, 4, "fp16"), # dp + tp + sp
    ]
)
def test_dense_ffn(bsz, hidden_size, intermediate_size, dp_attn, tp_attn, sp, dtype):
    reset_compute_graph()

    rank = 0
    stats = NodeStats()
    stats.new_iter(iter_id=0)

    seqlen = 1

    num_nodes = dp_attn * tp_attn * sp
    decode_cfg = SystemConfig().from_args(
        num_nodes=num_nodes, 
        dp_attn=dp_attn, # not relevant for FFN
        dp_ffn=1,
        tp_attn=tp_attn,
        tp_ffn=1,
        sp=sp,
        ep=num_nodes
    )

    for rank in range(num_nodes):
        dist_info = decode_cfg.get_dist_info(rank)
        dist_info.batch_mapping(bsz)
        batch_ids = dist_info.get_local_batchids("attn")
        local_bsz = len(batch_ids)

        op = FFN(f"{rank}ffn_0", hidden_size, intermediate_size, dist_info, dtype, is_dense_layer=True)

        x = Tensor("input", rank, [local_bsz, seqlen, hidden_size])

        y = op.forward(x, stats=stats)

        assert y.dims == [local_bsz, seqlen, hidden_size]

        par_factor = tp_attn * sp
        local_inter_size = intceil(intermediate_size / par_factor)
        ffn_comm_size = par_factor

        expected = {
            "memory_footprint": 3 * hidden_size * local_inter_size * dtype_to_byte(dtype), # weights only
            "num_ops": 3 * local_bsz * seqlen * hidden_size * local_inter_size, # in terms of MACs
            "hbm_reads": 3 * hidden_size * local_inter_size * dtype_to_byte(dtype), # weights only
            "network_data": 4 * intceil( (local_bsz * seqlen * hidden_size) / ffn_comm_size) * (ffn_comm_size -1) * dtype_to_byte(dtype) # allreduce for tp_ffn
        }

        op_mem_foot, op_num_ops, op_hbm_reads, op_net_data = op._stats.sumUp()
        assert expected["memory_footprint"] == op_mem_foot, f"Expected memory_footprint {expected['memory_footprint']}, got {op_mem_foot}"
        assert expected["num_ops"] == op_num_ops, f"Expected num_ops {expected['num_ops']}, got {op_num_ops}"
        assert expected["hbm_reads"] == op_hbm_reads, f"Expected hbm_reads {expected['hbm_reads']}, got {op_hbm_reads}"
        assert expected["network_data"] == op_net_data, f"Expected network_data {expected['network_data']}, got {op_net_data}"


if __name__=="__main__":
    test_ffn()