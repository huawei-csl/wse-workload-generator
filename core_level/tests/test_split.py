import pytest
import itertools

from core_level.layers.split import Split
from core_level.common.wafer import Wafer
from core_level.common.tensor import Tensor, reset_tensor_registry
from core_level.common.graph import Graph

@pytest.mark.parametrize(
    "split_dims,input_dims,tile_size,axis",
    [
        ([4, 4], [16, 8], [2, 2], 1), # even split
        ([8, 4], [12, 8], [2, 2], 0), # uneven split
        ([8, 24], [32, 16, 24], [8, 2, 6], 0), # 3-D tensor
    ]

)
def test_split(split_dims, input_dims, tile_size, axis):
    reset_tensor_registry()

    node_id = 1

    wafer = Wafer([4,4], [6,6])
    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:split_0"
        ops[node_id][op_id] = {
            "type": "Split",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:input_tensor_split{0}:{split_dims[0]}", f"{node_id}:input_tensor_split{split_dims[0]}:{split_dims[0]+split_dims[1]}"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    tensor_a = Tensor(f"{node_id}:input_tensor", input_dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

    layer = Split(f"{node_id}:split_0", node_id, axis, split_dims, input_dims, graph, "fp16")

    indices = [list(range(input_dims[i])) for i in range(len(input_dims))]
    for input_ind in itertools.product(*indices):
        out1 = tensor_a.get_physical_address([(d, d+1) for d in input_ind])

        if input_ind[axis] < split_dims[0]:
            output_ind = list(input_ind)

            out2 = layer.output_tensor0.get_physical_address([(e, e+1) for e in output_ind])
        else:
            output_ind = list(input_ind)
            output_ind[axis] = input_ind[axis] - split_dims[0]

            out2 = layer.output_tensor1.get_physical_address([(e, e+1) for e in output_ind])
        
        assert out1 == out2, "Mapping mismatch for input index ({}) vs output index ({}). Got {} vs {}.".format(input_ind, output_ind, out1, out2)
