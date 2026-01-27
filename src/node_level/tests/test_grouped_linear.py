import pytest
import json 

from src.node_level.layers.grouped_linear import GroupedLinear
from src.node_level.common.tensor import Tensor
from src.node_level.common.stats import NodeStats
from src.node_level.common.utils import dtype_to_byte
from src.node_level.common.compute_graph import reset_compute_graph


@pytest.mark.parametrize(
    "n_groups,batch_dims,in_features,out_features,dtype",
    [
        (4, [8], 32, 64, "fp16"), # 3D input case
        (8, [16, 4], 128, 256, "fp16"), # 4D input case
        (2, [16, 4], 128, 256, "fp8"), # fp8 case
    ]
)

def test_grouped_linear(n_groups, batch_dims, in_features, out_features, dtype):
    reset_compute_graph()

    rank = 0
    stats = NodeStats()
    stats.new_iter(iter_id=0)

    op = GroupedLinear("grouped_linear_0", rank, n_groups, in_features, out_features, dtype)

    x = Tensor("input", rank, [n_groups] + batch_dims + [in_features])
    out_tensor = op.forward(x, stats=stats)
    
    expected = {
        "memory_footprint": n_groups * in_features * out_features * dtype_to_byte(dtype), # fp16 = 2 bytes
        "num_ops": n_groups * eval("*".join(map(str, batch_dims))) * in_features * out_features, # in terms of MACs
        "hbm_reads": n_groups* in_features * out_features * dtype_to_byte(dtype) # weights only
    }

    assert out_tensor.dims == [n_groups] + batch_dims + [out_features], f"Output dims {out_tensor.dims} do not match expected {[n_groups] + batch_dims + [out_features]}"
    assert expected["memory_footprint"] == stats.get_stats("grouped_linear_0")["memory_footprint"], f"Memory footprint {stats.get_stats('grouped_linear_0')['memory_footprint']} does not match expected {expected['memory_footprint']}"
    assert expected["num_ops"] == stats.get_stats("grouped_linear_0")["num_ops"], f"Num ops {stats.get_stats('grouped_linear_0')['num_ops']} does not match expected {expected['num_ops']}"
    assert expected["hbm_reads"] == stats.get_stats("grouped_linear_0")["hbm_reads"], f"HBM reads {stats.get_stats('grouped_linear_0')['hbm_reads']} does not match expected {expected['hbm_reads']}"

if __name__=="__main__":
    test_grouped_linear(4, [8], 32, 64, "fp16")