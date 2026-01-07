import logging
import itertools
from typing import List

from core_level.common.tensor import Tensor

class View:
    def __init__(self, uid, node_id, input_dims, output_dims, graph, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.prec = prec

        assert eval("*".join([str(d) for d in input_dims])) == eval("*".join([str(d) for d in output_dims])), "View operation {} on node {} has incompatible dimensions: input_dims {} vs output_dims {}.".format(uid, node_id, input_dims, output_dims)
        
        self.input_dims = input_dims
        self.output_dims = output_dims

        first = -1
        for i in range(len(output_dims)):
            if input_dims[i] != output_dims[i]:
                first = i
                if input_dims[i] > output_dims[i]:
                    self.view_type = "split"
                    assert input_dims[:first] == output_dims[:first] and input_dims[first+1:] == output_dims[first+2:], "View operation support only splitting one dimension into two."
                    assert input_dims[first] == output_dims[first] * output_dims[first+1], "View operation {} on node {} has incompatible dimensions: input_dims {} vs output_dims {}.".format(uid, node_id, input_dims, output_dims)
                else:
                    self.view_type = "merge"
                    assert input_dims[:first] == output_dims[:first] and input_dims[first+2:] == output_dims[first+1:], "View operation support only merging two dimensions into one."
                    assert output_dims[first] == input_dims[first] * input_dims[first+1], "View operation {} on node {} has incompatible dimensions: input_dims {} vs output_dims {}.".format(uid, node_id, input_dims, output_dims)
                break

        self.graph_op = graph.get_op(node_id, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=input_dims,
            prec=self.prec,
        )
        assert self.input_tensor.tile_size is not None, "Input tensor {} of View operation {} on node {} does not have tile size.".format(self.input_tensor.uid, uid, node_id)

        if self.view_type == "split":
            # When one dim is split into two, the new tile size is 1 for the new dim.
            new_tile_size = self.input_tensor.tile_size[:first] + [1, self.input_tensor.tile_size[first]] + self.input_tensor.tile_size[first+1:]
        else:
            # When two dims are merged, take the tile size of the second dim.
            new_tile_size = self.input_tensor.tile_size[:first] + [self.input_tensor.tile_size[first+1]] + self.input_tensor.tile_size[first+2:]

        for d in range(len(self.output_dims)):
            assert self.output_dims[d] >= new_tile_size[d], "We do not support View operation with tile size larger than dimension size yet. View operation {} on node {} has output_dims {} with tile size {}.".format(uid, node_id, self.output_dims, new_tile_size)

        if self.view_type == "split":
            new_map = self._remap_split(self.input_tensor.memory_map, new_tile_size, first)
        else:
            new_map = self._remap_merge(self.input_tensor.memory_map, new_tile_size, first)

        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=output_dims,
            prec=self.prec,
        )
        self.output_tensor.set_map(new_map, new_tile_size, addr_offset=self.input_tensor.addr_offset)

    def _remap_merge(self, input_map, new_tile_size, first):
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

        new_map = {}

        tmp_dict = input_map
        old_indices = []
        for i in range(len(self.input_dims)):
            old_indices.append(list(tmp_dict.keys()))
            tmp_dict = tmp_dict[0]

        old_indices = list(itertools.product(*old_indices))

        for old_ind in old_indices:
            old_ind = list(old_ind)

            val = get_dict_val(input_map, old_ind)

            start_i, end_i = val["range"][first]
            start_j, end_j = val["range"][first+1]

            for i, row in enumerate(range(start_i, end_i)):
                new_ind = old_ind[:first] 
                new_ind += [(old_ind[first] * self.input_tensor.tile_size[first] + i) * (self.input_dims[first+1] // self.input_tensor.tile_size[first+1]) + old_ind[first+1]] 
                new_ind += old_ind[first+2:]

                new_range = val["range"][:first]
                
                new_start = row * self.input_dims[first+1] + start_j
                new_end = row * self.input_dims[first+1] + end_j

                new_range += [(new_start, new_end)]
                new_range += val["range"][first+2:]

                set_dict_val(
                    new_map, 
                    new_ind, 
                    {"range": new_range, "bank": val["bank"]}
                )

            logging.debug(f"Tile ind {old_ind} is mapped to {new_ind}.")

        return new_map


    def _remap_split(self, input_map, new_tile_size, first):
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

        new_rng = self.output_dims[first+1]//new_tile_size[first+1]
        
        new_map = {}

        tmp_dict = input_map
        old_indices = []
        for i in range(len(self.input_dims)):
            old_indices.append(list(tmp_dict.keys()))
            tmp_dict = tmp_dict[0]

        old_indices = list(itertools.product(*old_indices))

        for old_ind in old_indices:
            old_ind = list(old_ind)
            new_ind = old_ind[:first] + [old_ind[first] // new_rng, old_ind[first] % new_rng] + old_ind[first+1:]

            val = get_dict_val(input_map, old_ind)

            new_range = val["range"][:first]

            start_i, end_i = val["range"][first]
            new_range += [(start_i // self.output_dims[first+1], (end_i - 1) // self.output_dims[first+1] + 1)]
            new_range += [(start_i % self.output_dims[first+1], (end_i - 1) % self.output_dims[first+1] + 1)]

            new_range += val["range"][first+1:]

            set_dict_val(
                new_map, 
                new_ind, 
                {"range": new_range, "bank": val["bank"]}
            )

            logging.debug(f"Tile ind {old_ind} is mapped to {new_ind}.")

        return new_map



if __name__ == "__main__":
    from core_level.common.wafer import Wafer
    from core_level.common.tensor import Tensor, reset_tensor_registry
    from core_level.common.graph import Graph

    wafer = Wafer([4,4], [6,6])

    reset_tensor_registry()

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

    node_id = 1

    input_dims = [16, 32, 8, 8]
    output_dims = [16, 256, 8]

    tile_size = [4, 8, 4, 4]

    tensor_a = Tensor(f"{node_id}:input_tensor", input_dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

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
