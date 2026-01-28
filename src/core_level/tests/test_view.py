import pytest

from src.core_level.layers.view import View
from src.core_level.common.wafer import Wafer
from src.core_level.common.tensor import Tensor, reset_tensor_registry
from src.core_level.common.graph import Graph

@pytest.mark.parametrize(
    "input_dims,output_dims,tile_size",
    [
        ([8, 64, 8], [8, 1, 64, 8], [4, 4, 4]), # unsqueeze test
        ([8, 64, 8], [8, 64, 1, 8], [4, 4, 4]), # unsqueeze test
        ([8, 64, 8], [8, 4, 16, 8], [4, 4, 4]), # split dim test
        ([8, 64, 8], [8, 16, 4, 8], [4, 4, 4]), # split dim test
        ([8, 64, 8], [8, 16, 4, 8], [4, 32, 4]), # tile dim > new dim
    ]
)
def test_view_split(input_dims, output_dims, tile_size):
    reset_tensor_registry()

    wafer = Wafer([4,4], [6,6])

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:view_0"
        ops[node_id][op_id] = {
            "type": "View",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    node_id = 2

    tensor_a = Tensor(f"{node_id}:input_tensor", input_dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

    layer = View(f"{node_id}:view_0", node_id, input_dims, output_dims, graph, "fp16")

    assert tensor_a.get_mem_footprint() == layer.output_tensor.get_mem_footprint(), "Input and output tensor memory footprint do not match.".format()

    # check if mapping is exact
    for d0 in range(input_dims[0]):
        e0 = d0
        for d1 in range(input_dims[1]):
            e1 = d1 // layer.output_dims[2]
            e2 = d1 % layer.output_dims[2]
            for d2 in range(input_dims[2]):
                e3 = d2

                out1 = tensor_a.get_physical_address([(d0, d0+1), (d1, d1+1), (d2, d2+1)])
                out2 = layer.output_tensor.get_physical_address([(e0, e0+1), (e1, e1+1), (e2, e2+1), (e3, e3+1)])
                
                assert out1 == out2, "Mapping mismatch for input index ({},{},{}) vs output index ({},{},{},{}). Got {} vs {}.".format(d0, d1, d2, e0, e1, e2, e3, out1, out2)


@pytest.mark.parametrize(
    "input_dims,output_dims,tile_size",
    [
        ([4, 1, 8, 4], [4, 8, 4], [4, 1, 4, 4]), # squeeze dim test
        ([4, 8, 1, 4], [4, 8, 4], [4, 4, 1, 4]), # squeeze dim test
        ([4, 8, 8, 4], [4, 64, 4], [4, 4, 4, 4]), # merge dim test
        ([4, 4, 32, 4], [4, 128, 4], [4, 4, 4, 4]), # merge dim test
        ([4, 32, 4, 4], [4, 128, 4], [4, 4, 4, 4]), # merge dim test
    ]
)
def test_view_merge(input_dims, output_dims, tile_size):
    reset_tensor_registry()

    wafer = Wafer([4,4], [6,6])

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:view_0"
        ops[node_id][op_id] = {
            "type": "View",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    node_id = 3

    tensor_a = Tensor(f"{node_id}:input_tensor", input_dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=1)

    layer = View(f"{node_id}:view_0", node_id, input_dims, output_dims, graph, "fp16")

    assert tensor_a.get_mem_footprint() == layer.output_tensor.get_mem_footprint(), "Input and output tensor memory footprint do not match.".format()

    # check if mapping is exact
    for d0 in range(input_dims[0]):
        e0 = d0
        for d1 in range(input_dims[1]):
            for d2 in range(input_dims[2]):
                e1 = d1 * input_dims[2] + d2
                for d3 in range(input_dims[3]):
                    e2 = d3

                    out1 = tensor_a.get_physical_address([(d0, d0+1), (d1, d1+1), (d2, d2+1), (d3, d3+1)])
                    out2 = layer.output_tensor.get_physical_address([(e0, e0+1), (e1, e1+1), (e2, e2+1)])
                    
                    assert out1 == out2, "Mapping mismatch for input index ({},{},{},{}) vs output index ({},{},{}). Got {} vs {}.".format(d0, d1, d2, d3, e0, e1, e2, out1, out2)


if __name__=="__main__":
    test_view_split([64], [16, 4], [32])