import pytest 

from src.core_level.common.wafer import Wafer
from src.core_level.common.tensor import reset_tensor_registry
from src.core_level.common.graph import Graph
from src.core_level.common.tensor import Tensor
from src.core_level.layers.multicast import MulticastLayer

@pytest.mark.parametrize(
    "dims,tile_size,src_id,dst_ids", 
    [ 
        ([16, 16], [16, 16], 1, [2, ]), # no tiling, single dst
        ([16, 16], [16, 16], 1, [2, 3]), # no tiling, multi dst
        ([64, 64], [16, 16], 1, [2, 3]), # with tiling
        ([64, 64], [16, 16], 1, [0, 1, 2, 3]), # with tiling
    ]
)
def test_multicast(dims, tile_size, src_id, dst_ids):
    reset_tensor_registry()

    node_grid = (2, 2)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    ops = {}
    ops[src_id] = {}
    op_id = f"multicast_0"
    ops[src_id][op_id] = {
        "type": "Multicast",
        "inputs": [f"{src_id}:input_tensor"],
        "outputs": [f"{dst}:output_tensor" for dst in dst_ids]
    }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    input_tensor = Tensor(
        uid=f"{src_id}:input_tensor",
        dims=dims,
        prec="fp16",
    )
    input_tensor.map_to_memory(wafer.banks[src_id], tile_size=tile_size, addr_offset=0)

    layer = MulticastLayer(f"multicast_0", src_id, dst_ids, graph, dims, wafer, "fp16")

    expected = layer.calc_expected()
    assert expected["multicast"] == layer.stats.get_multicast(), f"Expected multicast bytes {expected['multicast']}, got {layer.stats.get_multicast()}"

if __name__=="__main__":
    test_multicast([16, 16], [16, 16], 1, [2, ])
