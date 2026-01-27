import pytest

from src.node_level.layers.linear import Linear
from src.node_level.common.tensor import Tensor
from src.node_level.common.stats import NodeStats
from src.node_level.common.utils import dtype_to_byte
from src.node_level.common.compute_graph import reset_compute_graph

@pytest.mark.parametrize(
    "batch_dims,in_features,out_features,dtype",
    [
        ([8], 32, 64, "fp16"), # 2D input case
        ([16, 4], 128, 256, "fp16"), # 3D input case
        ([16, 4], 128, 256, "fp8"), # fp8 case
    ]
)
def test_linear(batch_dims, in_features, out_features, dtype):
    reset_compute_graph()

    rank = 0
    stats = NodeStats()
    stats.new_iter(iter_id=0)

    op = Linear("linear_0", rank, in_features, out_features, dtype)

    x = Tensor("input", rank, batch_dims + [in_features])
    out_tensor = op.forward(x, stats=stats)

    expected = {
        "memory_footprint": in_features * out_features * dtype_to_byte(dtype), # fp16 = 2 bytes
        "num_ops": eval("*".join(map(str, batch_dims))) * in_features * out_features, # in terms of MACs
        "hbm_reads": in_features * out_features * dtype_to_byte(dtype) # weights only
    }

    assert out_tensor.dims == batch_dims + [out_features]
    assert expected["memory_footprint"] == stats.get_stats("linear_0")["memory_footprint"]
    assert expected["num_ops"] == stats.get_stats("linear_0")["num_ops"]
    assert expected["hbm_reads"] == stats.get_stats("linear_0")["hbm_reads"]

if __name__=="__main__":
    test_linear([16, 4], 128, 256, "fp16")