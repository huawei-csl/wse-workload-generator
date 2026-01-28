import logging
import itertools
from copy import deepcopy
from typing import List

from src.core_level.common.tensor import Tensor
from src.core_level.common.stats import Stats
from src.core_level.layers.remap import Remap

class Split:
    def __init__(self, uid, node_id, axis, split_dims, input_dims, graph, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.prec = prec

        assert len(split_dims) == 2, "Split operation supports splitting one dimension into two."
        assert input_dims[axis] == split_dims[0] + split_dims[1], "Split operation {} on node {} has incompatible dimensions: input_dims {} vs split_dims {}.".format(uid, node_id, input_dims, split_dims)
        assert -len(input_dims) <= axis < len(input_dims), "Split operation {} on node {} has invalid axis {} for input_dims {}.".format(uid, node_id, axis, input_dims)
        if axis < 0:
            axis += len(input_dims)
        
        self.axis = axis
        self.split_dims = split_dims
        self.input_dims = input_dims
        self.output_dims = deepcopy(input_dims)
        self.output_dims[axis] = split_dims

        self.graph_op = graph.get_op(node_id, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=input_dims,
            prec=self.prec,
        )
        assert self.input_tensor.tile_size is not None, "Input tensor {} of View operation {} on node {} does not have tile size.".format(self.input_tensor.uid, uid, node_id)

        if self.input_tensor.tile_size[axis] > split_dims[0] or self.input_tensor.tile_size[axis] > split_dims[1]:
            new_tile_size = list(self.input_tensor.tile_size)
            new_tile_size[axis] = min(split_dims)
            self.input_tensor = Remap(self.uid + "_remap", node_id, self.input_tensor, new_tile_size, wafer=None, prec=self.prec).get_output()

        assert split_dims[0] % self.input_tensor.tile_size[axis] == 0 and split_dims[1] % self.input_tensor.tile_size[axis] == 0, "We do not support Split operation with tile size larger than dimension size yet. Split operation {} on node {} has input_dims {} with tile size {}.".format(uid, node_id, input_dims, self.input_tensor.tile_size)

        new_dims0 = deepcopy(input_dims)
        new_dims0[axis] = split_dims[0]

        new_dims1 = deepcopy(input_dims)
        new_dims1[axis] = split_dims[1]

        new_map0, new_map1 = self._remap(self.input_tensor.memory_map)
        self.output_tensor0 = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=new_dims0,
            prec=self.prec,
        )
        self.output_tensor0.set_map(new_map0, self.input_tensor.tile_size, addr_offset=self.input_tensor.addr_offset)

        self.output_tensor1 = Tensor(
            uid=self.graph_op["outputs"][1],
            dims=new_dims1,
            prec=self.prec,
        )
        self.output_tensor1.set_map(new_map1, self.input_tensor.tile_size, addr_offset=self.input_tensor.addr_offset)

        self.stats = Stats()

    def _remap(self, input_map):
        def get_dict_val(dict, ind: List[int]):
            tmp_dict = dict
            for i in ind:
                tmp_dict = tmp_dict[i]
            return tmp_dict

        def set_dict_val(dict, ind: List[int], value):
            tmp_dict = dict
            for i in ind[:-1]:
                if i not in tmp_dict:
                    tmp_dict[i] = {}
                tmp_dict = tmp_dict[i]
            tmp_dict[ind[-1]] = value

        new_map0 = {}
        new_map1 = {}

        tmp_dict = input_map
        old_indices = []
        for i in range(len(self.input_dims)):
            old_indices.append(list(tmp_dict.keys()))
            tmp_dict = tmp_dict[0]

        old_indices = list(itertools.product(*old_indices))

        for old_ind in old_indices:
            old_ind = list(old_ind)

            val = get_dict_val(input_map, old_ind)

            if old_ind[self.axis] < self.split_dims[0] // self.input_tensor.tile_size[self.axis]:
                new_ind0 = list(old_ind)
                set_dict_val(
                    new_map0,
                    new_ind0,
                    val,
                )
            else:
                new_ind1 = list(old_ind)
                new_ind1[self.axis] = old_ind[self.axis] - (self.split_dims[0] // self.input_tensor.tile_size[self.axis])
                
                new_range = deepcopy(val["range"])
                new_range[self.axis] = (
                    val["range"][self.axis][0] - self.split_dims[0],
                    val["range"][self.axis][1] - self.split_dims[0],
                )
                set_dict_val(
                    new_map1,
                    new_ind1,
                    {
                        "range": new_range,
                        "bank": val["bank"],
                    },
                ) 

        return new_map0, new_map1
    
    def log_stats(self):
        return
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, dims=self.input_dims, tile_size=self.input_tensor.tile_size)


if __name__=="__main__":
    from src.core_level.common.wafer import Wafer
    from src.core_level.common.tensor import Tensor, reset_tensor_registry
    from src.core_level.common.graph import Graph

    wafer = Wafer([4,4], [6,6])

    reset_tensor_registry()

    node_id = 1

    input_dims = [4, 8]
    tile_size = [2, 2]
    split_dims = [4, 4]
    axis = 1

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