import pytest
import json 

from node_level.layers.allreduce import Allreduce
from node_level.common.tensor import Tensor
from stats import NodeStats
from utils import dtype_to_byte, intceil
from node_level.common.compute_graph import reset_compute_graph



@pytest.mark.parametrize(
    "dims,comm_group,dtype",
    [
        ([32,], [0], "fp16"), # single node (no communication)
        ([64, 64], [0,1,2,3], "fp16"), # 4 nodes
        ([128, 128], [0,1,2,3,4,5,6,7], "fp8"), # 8 nodes with fp8
        ([256, 256, 256], [0,1,2], "fp16"), # 3 nodes, not divisible case
    ]

)
def test_allreduce(dims, comm_group, dtype):
    reset_compute_graph()

    rank = 0
    stats = NodeStats()
    stats.new_iter(iter_id=0)

    op = Allreduce("linear_0", rank, dims, comm_group, dtype)

    x = Tensor("input", rank, dims)
    out_tensor = op.forward(x, stats=stats)

    comm_group_size = len(comm_group)
    vecsize = eval("*".join([str(d) for d in dims])) 
    expected = {
        # Assuming ring algorithm, each node sends and receives vecsize / len(comm_group) in each round
        # Both reduce and gather phases takes (len(comm_group) - 1) rounds
        # Thus, total network data per node = 4 * vecsize / len(comm_group) * (len(comm_group) - 1)  
        "network_data": 4 * intceil(vecsize / comm_group_size) * (comm_group_size -1) * dtype_to_byte(dtype)
    }

    assert out_tensor.dims == x.dims
    assert expected["network_data"] == stats.get_stats("linear_0")["network_data"]


if __name__=="__main__":
    test_allreduce([17, 17], [0, 1, 2], "fp16")