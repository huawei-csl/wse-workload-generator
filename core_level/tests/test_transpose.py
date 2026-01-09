import pytest
import itertools 

from core_level.layers.transpose import Transpose
from core_level.common.wafer import Wafer
from core_level.common.tensor import Tensor, reset_tensor_registry
from core_level.common.graph import Graph

@pytest.mark.parametrize(
    "input_dims,tile_size,trans_dims",
    [
        ([4, 6], [2, 2], [0, 1]), # 2D transpose
        ([4, 16, 9], [2, 2, 3], [2, 1]), # transpose in 3D matrix
    ]
)
def test_transpose(input_dims, tile_size, trans_dims):
    wafer = Wafer([4,4], [6,6])

    reset_tensor_registry()

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:transpose_0"
        ops[node_id][op_id] = {
            "type": "Transpose",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    node_id = 0
    output_dims = list(input_dims)
    output_dims[trans_dims[0]] = input_dims[trans_dims[1]]
    output_dims[trans_dims[1]] = input_dims[trans_dims[0]]

    tensor_a = Tensor(f"{node_id}:input_tensor", input_dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

    layer = Transpose(f"{node_id}:transpose_0", node_id, trans_dims, input_dims, output_dims, graph, "fp16")

    assert tensor_a.get_mem_footprint() == layer.output_tensor.get_mem_footprint(), "Input and output tensor memory footprint do not match.".format()

    indices = [list(range(input_dims[i])) for i in range(len(input_dims))]
    for input_ind in itertools.product(*indices):
        output_ind = list(input_ind)
        output_ind[trans_dims[0]] = input_ind[trans_dims[1]]
        output_ind[trans_dims[1]] = input_ind[trans_dims[0]]

        out1 = tensor_a.get_physical_address([(d, d+1) for d in input_ind])
        out2 = layer.output_tensor.get_physical_address([(e, e+1) for e in output_ind])
        
        assert out1 == out2, "Mapping mismatch for input index ({}) vs output index ({}). Got {} vs {}.".format(input_ind, output_ind, out1, out2)

if __name__== "__main__":
    test_transpose([4, 8, 6], [2, 2, 3], [1, 2])