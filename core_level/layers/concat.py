import logging
import itertools
from copy import deepcopy
from typing import List

from core_level.common.tensor import Tensor
from core_level.common.stats import Stats

class Concat:
    def __init__(self, uid, node_id, axis, input_dims, graph, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.prec = prec

        # input0_dims, input1_dims = input_dims
        assert -len(input_dims[0]) <= axis < len(input_dims[0]), "Concat operation {} on node {} has invalid axis {} for input_dims {}.".format(uid, node_id, axis, input0_dims)
        if axis < 0:
            axis += len(input_dims[0])

        # assert len(input_dims) == 2, "Concat operation {} on node {} requires two input tensors.".format(uid, node_id)
        for d in range(len(input_dims[0])):
            if d == axis:
                continue

            for i in range(1, len(input_dims)):
                assert input_dims[0][d] == input_dims[i][d], "Concat operation {} on node {} has incompatible dimensions: input0_dims {} vs input1_dims {}.".format(uid, node_id, input_dims[0], input_dims[i])

        self.input_dims = input_dims
        # self.input0_dims = input0_dims
        # self.input1_dims = input1_dims
        self.axis = axis

        self.graph_op = graph.get_op(node_id, uid)

        self.input_tensors = []
        for i in range(len(input_dims)):
            input_tensor = Tensor(
                uid=self.graph_op["inputs"][i],
                dims=input_dims[i],
                prec=self.prec,
            )
            assert input_tensor.tile_size is not None, "Input tensor {} of View operation {} on node {} does not have tile size.".format(input_tensor.uid, uid, node_id)
            self.input_tensors.append(input_tensor)
        
        for i in range(1, len(self.input_tensors)):
            assert self.input_tensors[0].tile_size == self.input_tensors[i].tile_size, "Concat operation {} on node {} requires input tensors to have the same tile size.".format(uid, node_id)

        # self.input_tensor0 = Tensor(
        #     uid=self.graph_op["inputs"][0],
        #     dims=input0_dims,
        #     prec=self.prec,
        # )
        # assert self.input_tensor0.tile_size is not None, "Input tensor {} of View operation {} on node {} does not have tile size.".format(self.input_tensor0.uid, uid, node_id)

        # self.input_tensor1 = Tensor(
        #     uid=self.graph_op["inputs"][1],
        #     dims=input1_dims,
        #     prec=self.prec,
        # )
        # assert self.input_tensor1.tile_size is not None, "Input tensor {} of View operation {} on node {} does not have tile size.".format(self.input_tensor0.uid, uid, node_id)

        # assert self.input_tensor0.tile_size == self.input_tensor1.tile_size, "Concat operation {} on node {} requires input tensors to have the same tile size.".format(uid, node_id)

        self.output_dims = deepcopy(input_dims[0])
        self.output_dims[axis] = sum([input_dims[i][axis] for i in range(len(input_dims))])

        out_map = self._remap([tensor.memory_map for tensor in self.input_tensors])

        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=self.output_dims,
            prec=self.prec,
        )
        self.output_tensor.set_map(out_map, self.input_tensors[0].tile_size)

        self.stats = Stats()

    def _remap(self, input_maps):
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

        # tmp_dict = input_map0
        # indices0 = []
        # for i in range(len(self.input0_dims)):
        #     indices0.append(list(tmp_dict.keys()))
        #     tmp_dict = tmp_dict[0]
        # indices0 = list(itertools.product(*indices0))

        # for ind0 in indices0:
        #     val = get_dict_val(input_map0, ind0)
        #     set_dict_val(new_map, list(ind0), val)

        # tmp_dict = input_map1
        # indices1 = []
        # for i in range(len(self.input1_dims)):
        #     indices1.append(list(tmp_dict.keys()))
        #     tmp_dict = tmp_dict[0]
        # indices1 = list(itertools.product(*indices1))

        # offset = self.input0_dims[self.axis] // self.input_tensor0.tile_size[self.axis]
        # for ind1 in indices1:
        #     new_ind1 = list(ind1)
        #     new_ind1[self.axis] += offset

        #     val = get_dict_val(input_map1, ind1)

        #     new_range = list(val["range"])
        #     new_range[self.axis] = (new_range[self.axis][0] + self.input0_dims[self.axis], new_range[self.axis][1] + self.input0_dims[self.axis])

        #     set_dict_val(
        #         new_map, 
        #         new_ind1, 
        #         {
        #             "range": new_range,
        #             "bank": val["bank"]
        #         }
        #     )


        addr_offset = 0

        for i in range(len(input_maps)):
            input_map = input_maps[i]
            tmp_dict = input_map
            indices = []
            for _ in range(len(self.input_dims[i])):
                indices.append(list(tmp_dict.keys()))
                tmp_dict = tmp_dict[0]
            indices = list(itertools.product(*indices))

            for ind in indices:
                new_ind = list(ind)
                new_ind[self.axis] += addr_offset // self.input_tensors[i].tile_size[self.axis]

                val = get_dict_val(input_map, ind)

                new_range = list(val["range"])
                new_range[self.axis] = (new_range[self.axis][0] + addr_offset, new_range[self.axis][1] + addr_offset)

                set_dict_val(
                    new_map, 
                    new_ind, 
                    {
                        "range": new_range,
                        "bank": val["bank"]
                    }
                )

            addr_offset += self.input_dims[i][self.axis]

        return new_map

    def log_stats(self):
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, dims=self.input_dims, tile_size=self.input_tensor0.tile_size)



if __name__=="__main__":
    from core_level.common.wafer import Wafer
    from core_level.common.tensor import Tensor, reset_tensor_registry
    from core_level.common.graph import Graph

    wafer = Wafer([4,4], [6,6])

    reset_tensor_registry()

    input0_dims = [8, 4, 2]
    input1_dims = [8, 2, 2]
    input2_dims = [8, 6, 2]
    tile_size = [4, 2, 1]
    axis = 1

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:concat_0"
        ops[node_id][op_id] = {
            "type": "Concat",
            "inputs": [f"{node_id}:input_tensor0", f"{node_id}:input_tensor1", f"{node_id}:input_tensor2"],
            "outputs": [f"{node_id}:output_tensor"]
        }

    node_id = 1

    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    tensor_0 = Tensor(f"{node_id}:input_tensor0", input0_dims, "fp16")
    tensor_0.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

    tensor_1 = Tensor(f"{node_id}:input_tensor1", input1_dims, "fp16")
    tensor_1.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

    tensor_2 = Tensor(f"{node_id}:input_tensor2", input2_dims, "fp16")
    tensor_2.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=0)

    layer = Concat(f"{node_id}:concat_0", node_id, axis, [input0_dims, input1_dims, input2_dims], graph, "fp16")

    indices = [list(range(input0_dims[i])) for i in range(len(input0_dims))]
    for input_ind in itertools.product(*indices):
        output_ind = list(input_ind)

        out1 = tensor_0.get_physical_address([(d, d+1) for d in input_ind])
        out2 = layer.output_tensor.get_physical_address([(e, e+1) for e in output_ind])

        assert out1 == out2, "Mapping mismatch for input index ({}) vs output index ({}). Got {} vs {}.".format(input_ind, output_ind, out1, out2)

    indices = [list(range(input1_dims[i])) for i in range(len(input1_dims))]
    for input_ind in itertools.product(*indices):
        output_ind = list(input_ind)
        output_ind[axis] += input0_dims[axis]

        out1 = tensor_1.get_physical_address([(d, d+1) for d in input_ind])
        out2 = layer.output_tensor.get_physical_address([(e, e+1) for e in output_ind])

        assert out1 == out2, "Mapping mismatch for input index ({}) vs output index ({}). Got {} vs {}.".format(input_ind, output_ind, out1, out2)

    indices = [list(range(input2_dims[i])) for i in range(len(input2_dims))]
    for input_ind in itertools.product(*indices):
        output_ind = list(input_ind)
        output_ind[axis] += (input0_dims[axis] + input1_dims[axis])

        out1 = tensor_2.get_physical_address([(d, d+1) for d in input_ind])
        out2 = layer.output_tensor.get_physical_address([(e, e+1) for e in output_ind])

        assert out1 == out2, "Mapping mismatch for input index ({}) vs output index ({}). Got {} vs {}.".format(input_ind, output_ind, out1, out2)