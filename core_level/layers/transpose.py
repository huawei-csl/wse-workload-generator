import logging
import itertools
from copy import deepcopy

from typing import List

from core_level.common.tensor import Tensor
from core_level.common.stats import Stats

class Transpose:
    def __init__(self, uid, node_id, axes, input_dims, output_dims, graph, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.axes = axes
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.prec = prec

        assert len(axes) == 2, "Transpose supports transposing only two dimensions."
        assert eval("*".join([str(d) for d in input_dims])) == eval("*".join([str(d) for d in output_dims])), "Transpose operation {} on node {} has incompatible dimensions: input_dims {} vs output_dims {}.".format(uid, node_id, input_dims, output_dims)
        assert input_dims[axes[0]] == output_dims[axes[1]] and input_dims[axes[1]] == output_dims[axes[0]], "Transpose operation {} on node {} has incompatible dimensions: axes: {} input_dims {} vs output_dims {}.".format(uid, node_id, trans_dims, input_dims, output_dims)

        self.graph_op = graph.get_op(node_id, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=input_dims,
            prec=self.prec,
        )

        td0, td1 = axes
        new_tile_size = list(self.input_tensor.tile_size)
        new_tile_size[td0] = self.input_tensor.tile_size[td1]
        new_tile_size[td1] = self.input_tensor.tile_size[td0]

        new_map = self.remap()

        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=output_dims,
            prec=self.prec,
        )
        self.output_tensor.set_map(new_map, new_tile_size, addr_offset=self.input_tensor.addr_offset)

        self.stats = Stats()

    def remap(self):
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

        td0, td1 = self.axes
        new_map = {}

        tmp_dict = self.input_tensor.memory_map
        old_indices = []
        for i in range(len(self.input_dims)):
            old_indices.append(list(tmp_dict.keys()))
            tmp_dict = tmp_dict[0]

        old_indices = list(itertools.product(*old_indices))

        for old_ind in old_indices:
            new_ind = list(old_ind)
            new_ind[td0], new_ind[td1] = old_ind[td1], old_ind[td0]

            val = get_dict_val(self.input_tensor.memory_map, list(old_ind))
            new_range = list(val["range"])
            new_range[td0], new_range[td1] = val["range"][td1], val["range"][td0]

            set_dict_val(
                new_map, 
                new_ind, 
                {
                    "range": new_range,
                    "bank": val["bank"]
                }
            )
        
        return new_map


    def log_stats(self):
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, dims=self.input_dims, tile_size=self.input_tensor.tile_size)


if __name__== "__main__":
    from core_level.common.wafer import Wafer
    from core_level.common.tensor import Tensor, reset_tensor_registry
    from core_level.common.graph import Graph

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

    input_dims = [4, 8, 6]
    tile_size = [2, 2, 3]
    node_id = 0
    axes = [1, 2]
    output_dims = list(input_dims)
    output_dims[axes[0]] = input_dims[axes[1]]
    output_dims[axes[1]] = input_dims[axes[0]]

    tensor_a = Tensor(f"{node_id}:input_tensor", input_dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

    layer = Transpose(f"{node_id}:transpose_0", node_id, axes, input_dims, output_dims, graph, "fp16")

    assert tensor_a.get_mem_footprint() == layer.output_tensor.get_mem_footprint(), "Input and output tensor memory footprint do not match.".format()

    for d0 in range(input_dims[0]):
        for d1 in range(input_dims[1]):
            for d2 in range(input_dims[2]):
                e0, e1, e2 = d0, d2, d1
                
                out1 = tensor_a.get_physical_address([(d0, d0+1), (d1, d1+1), (d2, d2+1)])
                out2 = layer.output_tensor.get_physical_address([(e0, e0+1), (e1, e1+1), (e2, e2+1)])
                
                assert out1 == out2, "Mapping mismatch for input index ({},{},{}) vs output index ({},{},{}). Got {} vs {}.".format(d0, d1, d2, e0, e1, e2, out1, out2)
