
import pytest 

from src.core_level.common.wafer import Wafer
from src.core_level.common.tensor import reset_tensor_registry
from src.core_level.common.graph import Graph
from src.core_level.common.tensor import Tensor
from src.core_level.layers.allreduce import AllreduceLayer

@pytest.mark.parametrize(
    "dims, tile_size, comm_group",
    [
        # allreduce across 1 nodes -> no op
        (
            [32, 64],
            [8, 8],
            [0],
        ),
        # allreduce across 2 nodes, no tiling
        (
            [32, 64],
            [32, 64],
            [0, 1],
        ),
        # allreduce across 4 nodes, with tiling
        (
            [128, 64],
            [16, 4],
            [0, 1, 2, 3],
        ),
        # 3-dimensional
        (
            [128, 32, 64],
            [16, 8, 4],
            [0, 1, 2, 3],
        ),
    ]
)

def test_allreduce(dims, tile_size, comm_group):
    reset_tensor_registry()

    node_grid = (2, 2)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"allreduce_0"
        ops[node_id][op_id] = {
            "type": "Allreduce",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    for node_id in comm_group:
        input_tensor = Tensor(
            uid=f"{node_id}:input_tensor",
            dims=dims,
            prec="fp16",
        )
        input_tensor.map_to_memory(wafer.banks[node_id], tile_size=tile_size, addr_offset=0)

    for node_id in comm_group:
        layer = AllreduceLayer(f"allreduce_0", node_id, comm_group, graph, dims, wafer, "fp16")
        expected = layer.calc_expected()
        assert expected["copy"] == layer.stats.get_copy(), "AllreduceLayer stats do not match expected values."
        assert expected["reads"] == layer.stats.get_reads(), "AllreduceLayer stats do not match expected values."
        assert expected["writes"] == layer.stats.get_writes(), "AllreduceLayer stats do not match expected values."
        assert expected["vector_flops"] == layer.stats.get_vector(), "AllreduceLayer stats do not match expected values."

if __name__=="__main__":
    test_allreduce(dims=[64, 32], tile_size=[16, 16], comm_group=[0, 1, 2, 3])