import pytest
import itertools

from src.core_level.layers.slice import Slice
from src.core_level.common.wafer import Wafer
from src.core_level.common.tensor import Tensor, reset_tensor_registry
from src.core_level.common.graph import Graph

@pytest.mark.parametrize(
    "index_rng,input_dims,tile_size,axis",
    [
        ([0, 1], [8, 8], [2, 2], 0), # single element slice along axis 0
        ([4, 8], [16, 16], [4, 4], 0), # larger tensor slice along axis 0
        ([8, 14], [8, 16, 8], [4, 4, 4], 1), # 3-D tensor slice along axis 1
        ([5, 6], [8, 16, 8], [4, 4, 4], 1), # Single element, not aligned to tile size
    ]
)
def test_slice(index_rng, input_dims, tile_size, axis):
    reset_tensor_registry()

    node_id = 1

    wafer = Wafer([4,4], [6,6])

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:slice_0"
        ops[node_id][op_id] = {
            "type": "Slice",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:input_tensor_slice{index_rng[0]}:{index_rng[1]}"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    tensor_a = Tensor(f"{node_id}:input_tensor", input_dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

    layer = Slice(f"{node_id}:slice_0", node_id, axis, index_rng, input_dims, graph, "fp16")

    indices = [list(range(input_dims[i])) for i in range(len(input_dims))]
    for input_ind in itertools.product(*indices):

        if input_ind[axis] >= index_rng[0] and input_ind[axis] < index_rng[1]: 
            output_ind = list(input_ind)
            output_ind[axis] = input_ind[axis] - index_rng[0]

            out1 = tensor_a.get_physical_address([(d, d+1) for d in input_ind])
            out2 = layer.output_tensor.get_physical_address([(e, e+1) for e in output_ind])
            assert out1 == out2, "Mapping mismatch for input index ({}) vs output index ({}). Got {} vs {}.".format(input_ind, output_ind, out1, out2)
