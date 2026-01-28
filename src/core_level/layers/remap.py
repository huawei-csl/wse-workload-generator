import itertools

from typing import List
from src.node_level.common.utils import intceil

from src.core_level.common.tensor import Tensor
from src.core_level.layers.unicast import TileUnicastOp
from src.core_level.common.stats import Stats

class Remap:
    def __init__(self, uid, node_id, input_tensor, new_tile_size, wafer, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.new_tile_size = new_tile_size
        self.wafer = wafer
        self.prec = prec

        assert input_tensor.memory_map is not None, "Input tensor {} of Remap operation {} on node {} does not have memory map.".format(input_tensor.uid, uid, node_id)

        for i in range(len(input_tensor.tile_size)):
            assert  new_tile_size[i] <= input_tensor.tile_size[i], "Remap operation {} on node {} has invalid new tile size {} for input tensor tile size {}.".format(uid, node_id, new_tile_size, input_tensor.tile_size)


        self.input_tensor = input_tensor
        self.dims = input_tensor.dims

        self.output_tensor = Tensor(
            uid=f"{input_tensor.uid}_remap",
            dims=list(self.dims),
            prec=self.prec,
        )

        self.stats = Stats()

        self.copy()

    def copy(self):
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

        indices = [list(range(intceil(self.dims[i]/self.new_tile_size[i]))) for i in range(len(self.dims))]

        in_map = self.input_tensor.memory_map
        out_map = {}

        for ind in list(itertools.product(*indices)):
            starts = [ind[i] * self.new_tile_size[i]  for i in range(len(ind))]
            ends = [min(starts[i] + self.new_tile_size[i], self.dims[i]) for i in range(len(ind))]
            rng = [(starts[i], ends[i]) for i in range(len(ind))]

            physical_addresses = self.input_tensor.get_physical_address(rng)
            assert len(physical_addresses) == 1, "Remap operation {} on node {} requires input tile to be mapped to a single physical bank, got {} banks for tile range {}.".format(self.uid, self.node_id, len(physical_addresses), rng)

            # largest_bank = max(physical_addresses, key=lambda k: physical_addresses[k])

            val = {
                "range": rng,
                "bank": list(physical_addresses.keys())[0],
            }

            set_dict_val(out_map, ind, {"range": val["range"], "bank": list(physical_addresses.keys())[0]})

        self.output_tensor.set_map(out_map, self.new_tile_size)

        # core_id = 0
        # for ind in list(itertools.product(*indices)):
        #     val = get_dict_val(out_map, ind)
        #     physical_addresses = self.input_tensor.get_physical_address(val["range"])

        #     for src_bank, data_size in physical_addresses.items():
        #         if src_bank == val["bank"]:
        #             continue
                
        #         in_tile = self.input_tensor.slice(val["range"])
        #         out_tile = self.output_tensor.slice(val["range"])
        #         copy_op = TileUnicastOp("{}_unicast_{}_src{}".format(self.uid, "_".join([str(i) for i in ind]), src_bank.bank_id), in_tile, out_tile)
        #         core = self.wafer.get_core(self.node_id, core_id)
        #         copy_op.map_to_core(core)
        #         self.stats.merge(copy_op.stats)
        #         core_id += 1

    def get_output(self):
        return self.output_tensor
    
    def log_stats(self):
        expected = None
        self.stats.log_stats(self.uid, self.__class__.__name__, self.src, expected=expected, dims=self.dims, tile_size=self.tile_size)

if __name__=="__main__":
    from src.core_level.common.wafer import Wafer
    from src.core_level.common.tensor import reset_tensor_registry

    reset_tensor_registry()

    node_grid = (1, 1)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    dims = [147, 1, 128, 192]
    tile_size = [147, 1, 1, 192]
    new_tile_size = [147, 1, 1, 64]

    node_id = 0

    input_tensor = Tensor(
        uid=f"{node_id}:input_tensor",
        dims=dims,
        prec="fp16",
    )
    input_tensor.map_to_memory(wafer.banks[node_id], tile_size=tile_size, addr_offset=0)

    op = Remap("remap_0", node_id, input_tensor, new_tile_size, wafer=wafer, prec="fp16")
    print(op.stats.data)