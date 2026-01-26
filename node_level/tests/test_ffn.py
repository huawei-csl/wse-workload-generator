
import pytest 

from stats import NodeStats
from utils import dtype_to_byte, intceil
from config import SystemConfig

from node_level.layers.ffn import FFN
from node_level.common.tensor import Tensor
from node_level.common.compute_graph import reset_compute_graph

@pytest.mark.parametrize(
    "bsz,hidden_size,intermediate_size,tp_ffn,ep,is_dense_layer,dtype",
    [
        (4, 1024, 512, 1, 1, False, "fp16"), # single node
        (4, 1024, 512, 2, 1, False, "fp16"), # tp_ffn = 2
        (8, 1024, 512, 3, 1, False, "fp16"), # tp_ffn = 3 uneven case
        (16, 2048, 1024, 4, 1, False, "fp8"), # tp_ffn = 4 with fp8
        (4, 1024, 512, 1, 4, True, "fp8"), # dense layer case
    ]
)
def test_ffn(bsz, hidden_size, intermediate_size, tp_ffn, ep, is_dense_layer, dtype):
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

        op = FFN(f"{rank}ffn_0", hidden_size, intermediate_size, dist_info, dtype, is_dense_layer=is_dense_layer)

        x = Tensor("input", rank, [local_bsz, seqlen, hidden_size])

        y = op.forward(x, stats=stats)

        assert y.dims == [local_bsz, seqlen, hidden_size]

        par_factor = tp_ffn if not is_dense_layer else ep
        comm_group = dist_info.ffn_comm_groups["tp_ffn"] if not is_dense_layer else dist_info.ffn_comm_groups["ep"]
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

if __name__=="__main__":
    test_ffn()