import logging
import itertools
from copy import deepcopy
from typing import List

from core_level.common.tensor import Tensor
from core_level.common.stats import Stats


class Slice:
    def __init__(self, uid, node_id, axis, index_rng, input_dims, graph, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.prec = prec

        assert -len(input_dims) <= axis < len(input_dims), "Slice operation {} on node {} has invalid axis {} for input_dims {}.".format(uid, node_id, axis, input_dims)
        if axis < 0:
            axis += len(input_dims)

        assert len(index_rng) == 2, "Index range should be a tuple of (start, end)." 
        assert index_rng[0] >= 0 and index_rng[1] <= input_dims[axis], "Slice operation {} on node {} has invalid index range {} for axis {} with input_dims {}.".format(uid, node_id, index_rng, axis, input_dims)
        assert index_rng[1] > index_rng[0], "Slice operation {} on node {} has invalid index range {}.".format(uid, node_id, index_rng)

        self.axis = axis
        self.index_rng = index_rng
        self.input_dims = input_dims

        self.graph_op = graph.get_op(node_id, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=input_dims,
            prec=self.prec,
        )
        assert self.input_tensor.tile_size is not None, "Input tensor {} of View operation {} on node {} does not have tile size.".format(self.input_tensor.uid, uid, node_id)
        assert index_rng[0] % self.input_tensor.tile_size[axis] == 0, "Slice operation supports index range aligned with tile size."

        new_map = self._remap(self.input_tensor.memory_map)

        self.output_dims = deepcopy(input_dims)
        self.output_dims[axis] = index_rng[1] - index_rng[0]
        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=self.output_dims,
            prec=self.prec,
        )

        new_tile_size = list(self.input_tensor.tile_size)
        new_tile_size[axis] = min(self.input_tensor.tile_size[axis], self.output_dims[axis])
        self.output_tensor.set_map(new_map, new_tile_size, addr_offset=self.input_tensor.addr_offset)

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

        start_idx, end_idx = self.index_rng

        new_map = {}

        tmp_dict = input_map
        old_indices = []
        for i in range(len(self.input_dims)):
            old_indices.append(list(tmp_dict.keys()))
            tmp_dict = tmp_dict[0]

        old_indices = list(itertools.product(*old_indices))

        for old_ind in old_indices:
            val = get_dict_val(input_map, old_ind)

            if old_ind[self.axis] < start_idx // self.input_tensor.tile_size[self.axis]:
                continue

            if old_ind[self.axis] > (end_idx - 1) // self.input_tensor.tile_size[self.axis]:
                continue

            new_ind = list(old_ind)
            new_ind[self.axis] = old_ind[self.axis] - (start_idx // self.input_tensor.tile_size[self.axis])
            
            new_range = deepcopy(val["range"])
            new_range[self.axis] = (
                val["range"][self.axis][0] - start_idx, 
                min(val["range"][self.axis][1] - start_idx, end_idx - start_idx)
            )
            
            set_dict_val(
                new_map,
                new_ind,
                {
                    "bank": val["bank"],
                    "range": new_range,
                }
            )
        return new_map
    
    def log_stats(self):
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, dims=self.input_dims, tile_size=self.input_tensor.tile_size)


if __name__=="__main__":
    from core_level.common.wafer import Wafer
    from core_level.common.tensor import Tensor, reset_tensor_registry
    from core_level.common.graph import Graph

    wafer = Wafer([4,4], [6,6])

    reset_tensor_registry()

    node_id = 1

    input_dims = [4, 8]
    tile_size = [2, 2]
    index_rng = [3, 5]
    axis = 1

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