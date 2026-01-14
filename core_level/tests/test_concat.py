import pytest 
import itertools

from core_level.layers.concat import Concat

@pytest.mark.parametrize(
    "input_dims,axis,tile_size",
    [
        ([[4, 8], [4, 4]], 1, [2, 2]),         # 2-tensor concat 2-D tensor
        ([[8, 4, 2], [8, 2, 2]], 1, [4, 2, 1]), # 2-tensor concat 3-D tensor
        ([[8, 4, 2], [8, 2, 2], [8, 6, 2]], 1, [4, 2, 1]), # 3-tensor concat 3-D tensor
    ]
)
def test_concat(input_dims, axis, tile_size):
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
            "inputs": [f"{node_id}:input{i}_tensor" for i in range(len(input_dims))],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    node_id = 0

    input_tensors = []
    for i in range(len(input_dims)):
        tensor = Tensor(f"{node_id}:input{i}_tensor", input_dims[i], "fp16")
        tensor.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=1)
        input_tensors.append(tensor)

    output_dims = list(input_dims[0])
    output_dims[axis] = sum([input_dims[i][axis] for i in range(len(input_dims))])

    layer = Concat(f"{node_id}:concat_0", node_id, axis, input_dims, graph, "fp16")

    addr_offset = 0
    for i in range(len(input_dims)):
        indices = [list(range(input_dims[i][j])) for j in range(len(input_dims[i]))]
        for input_ind in itertools.product(*indices):
            output_ind = list(input_ind)        
            output_ind[axis] += addr_offset

            out1 = input_tensors[i].get_physical_address([(d, d+1) for d in input_ind])
            out2 = layer.output_tensor.get_physical_address([(e, e+1) for e in output_ind])
            
            assert out1 == out2, "Mapping mismatch for input0 index ({}) vs output index ({}). Got {} vs {}".format(input_ind, output_ind, out1, out2)
        addr_offset += input_dims[i][axis]
