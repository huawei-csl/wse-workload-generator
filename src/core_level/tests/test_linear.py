import logging 
import pytest
from src.node_level.common.utils import byte_to_str

from src.core_level.layers.linear import LinearLayer
from src.core_level.common.wafer import Wafer
from src.core_level.common.graph import Graph
from src.core_level.common.isa import InstructionSet
from src.core_level.common.tensor import reset_tensor_registry

@pytest.mark.parametrize(
    "node_grid,core_grid,dims,tile_size", 
    [
        # single node, no tiling test case
        (
            [1, 1], 
            [4, 4],
            [16, 16, 16], 
            [16, 16, 16], 
        ),
        # multi-node, with tiling test case, no split-K
        (
            [2, 2], 
            [4, 4],
            [32, 16, 32], 
            [16, 16, 16], 
        ),
        # multi-node, with tiling test case, with split-K
        (
            [4, 4], 
            [4, 4],
            [32, 32, 32], 
            [16, 16, 16], 
        ),
    ]
)
def test_linear(node_grid, core_grid, dims, tile_size):
    reset_tensor_registry()

    wafer = Wafer(node_grid, core_grid)

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:linear_0"
        ops[node_id][op_id] = {
            "type": "Linear",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    for node_id in range(wafer.num_nodes):
        layer = LinearLayer(f"{node_id}:linear_0", node_id, graph, dims, tile_size, wafer, prec="fp16")

    M, K, N = layer.dims 
    Tm, Tk, Tn = layer.tile_size
    expected = layer.calc_expected()

    if K == Tk:
        # no split-K
        total_reads = expected["input0_size"] * (N // Tn) + expected["input1_size"] * (M // Tm)
        total_writes = expected["output_size"]
    else:
        total_reads = expected["input0_size"] * (N // Tn) + expected["input1_size"] * (M // Tm) 
        total_reads += expected["output_size"]  * (K // Tk) # needs to read partial sums
        total_writes = expected["output_size"] * (K // Tk + 1)  # needs to write partial sums

    assert total_reads == layer.stats.get_reads(), f"Expected {expected['reads']} reads, got {layer.stats.get_reads()}."
    assert total_writes == layer.stats.get_writes(), f"Expected {expected['writes']} writes, got {layer.stats.get_writes()}."
    assert expected["flops"] == layer.stats.get_total_cube(), f"Expected {expected['flops']} flops, got {layer.stats.get_total_cube()}."

if __name__=="__main__":
    node_grid = [1, 1]
    core_grid = [8, 8]
    dims = [16, 128, 576]
    tile_size = [4, 16, 64]

    test_linear(
        node_grid,
        core_grid,
        dims, 
        tile_size
    )