
from typing import List
import logging 
from copy import deepcopy

from src.core_level.common.isa import InstructionSet
from src.core_level.common.stats import Stats
from src.core_level.common.tensor import Tensor

class TileReduceOp:
    def __init__(self, id, in_tiles, out_tile) -> None:
        self.id = id
        self.in_tiles = in_tiles
        for tile in self.in_tiles[1:]:
            assert tile.dims == in_tiles[0].dims, "Input tiles must have the same dimensions in TileReduceOp {}.".format(self.id)

        self.out_tile = out_tile
        assert out_tile.dims == in_tiles[0].dims, "Output tile must have the same dimensions as input tiles in TileReduceOp {}.".format(self.id)

        self.mapped_core = None
        self.stats = Stats()
        logging.debug("TileReduceOp {} is created with in_tiles {}, out tile {}.".format(self.id, [t.id for t in self.in_tiles], self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileReduceOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileReduceOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))
    
    def get_traces(self) -> List[str]:
        traces = []

        # Read input tiles from memory
        for i, tile in enumerate(self.in_tiles):
            mem_sizes = tile.get_physical_address()
            for bank, size in mem_sizes.items():
                traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
                self.stats.add_reads(size)

            if i > 0:
                traces.append(InstructionSet.ADD(self.out_tile.dims, self.id))
                self.stats.add_vector(self.mapped_core.core_id, eval("*".join(map(str, self.out_tile.dims))))

        # Write output tile back to memory
        mem_sizes = self.out_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.WRITE(bank.bank_id, size, self.id))
            self.stats.add_writes(size)

        return traces

class Sum:
    def __init__(self, uid, node_id, axis, input_dims, graph, wafer, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.wafer = wafer
        self.prec = prec

        assert -len(input_dims) <= axis < len(input_dims), "Sum operation {} on node {} has invalid axis {} for input_dims {}.".format(uid, node_id, axis, input_dims)
        if axis < 0:
            axis += len(input_dims)

        self.axis = axis
        self.input_dims = input_dims

        self.graph_op = graph.get_op(node_id, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=input_dims,
            prec=self.prec,
        )
        assert self.input_tensor.tile_size is not None, "Input tensor {} of View operation {} on node {} does not have tile size.".format(self.input_tensor.uid, uid, node_id)

        self.output_dims = deepcopy(input_dims)
        self.output_dims[axis] = 1

        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=self.output_dims,
            prec=self.prec,
        )

        new_tile_size = list(self.input_tensor.tile_size)
        new_tile_size[axis] = 1
        self.out_tile_size = new_tile_size

        self.output_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=self.out_tile_size, addr_offset=0)
        
        core_ids = [self.wafer.get_core(self.node_id, i).core_id for i in range(self.wafer.num_cores_per_node)]
        self.stats = Stats(core_ids)

        self.create_tiles()
        self.create_ops()

        self.map_ops()

    def create_tiles(self):
        D0, D1, D2 = self.input_dims
        T0, T1, T2 = self.input_tensor.tile_size

        self.input_tiles= {}
        self.output_tiles = {}
        if self.axis == 0:
            for i1, p1 in enumerate(range(0, D1, T1)):
                self.input_tiles[i1] = {}
                self.output_tiles[i1] = {}
                for i2, p2 in enumerate(range(0, D2, T2)):
                    self.input_tiles[i1][i2] = []
                    for i0, _ in enumerate(range(0, D0, 1)):
                        tiled_1 = min(T1, D1 - p1)
                        tiled_2 = min(T2, D2 - p2)

                        self.input_tiles[i1][i2].append(self.input_tensor.slice([(i0, i0 + 1), (i1*T1, i1*T1 + tiled_1), (i2*T2, i2*T2 + tiled_2)]))
                    self.output_tiles[i1][i2] = self.output_tensor.slice([(0, 1), (i1*T1, i1*T1 + tiled_1), (i2*T2, i2*T2 + tiled_2)])
        else:
            NotImplementedError

    def create_ops(self):
        self.reduce_ops = {}
        for i1 in self.input_tiles:
            self.reduce_ops[i1] = {}
            for i2 in self.input_tiles[i1]:
                self.reduce_ops[i1][i2] = TileReduceOp("{}_reduce_op".format(self.uid), self.input_tiles[i1][i2], self.output_tiles[i1][i2])

    def map_ops(self):
        core_idx = 0
        for i1 in self.reduce_ops:
            for i2 in self.reduce_ops[i1]:
                core = self.wafer.get_core(self.node_id, core_idx)
                self.reduce_ops[i1][i2].map_to_core(core)
                core_idx = (core_idx + 1) % self.wafer.num_cores_per_node

    def calc_expected(self):
        expected = {
            "flops": eval("*".join(map(str, self.input_dims))),
            "reads": eval("*".join(map(str, self.input_dims))),
            "writes": eval("*".join(map(str, self.output_dims))),
        }
        return expected
    
    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, expected=expected, dims=self.input_tensor.dims, tile_size=self.input_tensor.tile_size)
