import pytest 
import itertools

from core_level.layers.concat import Concat

@pytest.mark.parametrize(
    "input0_dims,input1_dims,axis,tile_size",
    [
        ([4, 8], [4, 4], 1, [2, 2]),         # concat 2-D tensor
        ([8, 4, 2], [8, 2, 2], 1, [4, 2, 1]), # concat 3-D tensor
    ]
)
def test_concat(input0_dims, input1_dims, axis, tile_size):
    from core_level.common.wafer import Wafer
    from core_level.common.tensor import Tensor, reset_tensor_registry
    from core_level.common.graph import Graph

    wafer = Wafer([4,4], [6,6])

    reset_tensor_registry()

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:concat_0"
        ops[node_id][op_id] = {
            "type": "Concat",
            "inputs": [f"{node_id}:input0_tensor", f"{node_id}:input1_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    node_id = 0

    tensor_a = Tensor(f"{node_id}:input0_tensor", input0_dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=1)

    tensor_b = Tensor(f"{node_id}:input1_tensor", input1_dims, "fp16")
    tensor_b.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=3)

    output_dims = list(input0_dims)
    output_dims[axis] += input1_dims[axis]

    layer = Concat(f"{node_id}:concat_0", node_id, axis, [input0_dims, input1_dims], graph, "fp16")

    # check if mapping is exact
    indices0 = [list(range(input0_dims[i])) for i in range(len(input0_dims))]
    for input_ind in itertools.product(*indices0):
        output_ind = list(input_ind)

        out1 = tensor_a.get_physical_address([(d, d+1) for d in input_ind])
        out2 = layer.output_tensor.get_physical_address([(e, e+1) for e in output_ind])
        
        assert out1 == out2, "Mapping mismatch for input0 index ({}) vs output index ({}). Got {} vs {}".format(input_ind, output_ind, out1, out2)

    indices1 = [list(range(input1_dims[i])) for i in range(len(input1_dims))]
    for input_ind in itertools.product(*indices1):
        output_ind = list(input_ind)
        output_ind[axis] += input0_dims[axis]

        out1 = tensor_b.get_physical_address([(d, d+1) for d in input_ind])
        out2 = layer.output_tensor.get_physical_address([(e, e+1) for e in output_ind])
        
        assert out1 == out2, "Mapping mismatch for input1 index ({}) vs output index ({}). Got {} vs {}".format(input_ind, output_ind, out1, out2)