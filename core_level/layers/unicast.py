import logging

from typing import List
from utils import intceil
from core_level.common.tile import Tile
from core_level.common.graph import get_compute_graph

from core_level.common.tensor import Tensor

class TileUnicastOp:
    def __init__(self, id, input_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.out_tile = out_tile
        self.mapped_core = None
        logging.debug("TileUnicastOp {} is created with input tile {}, out tile {}.".format(self.id, self.input_tile.id, self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileUnicastOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileUnicastOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        traces = []
        traces.append("COPY {} {} {} {} {}".format(self.input_tile.id, self.input_tile.mem_bank.bank_id, self.out_tile.id, self.out_tile.mem_bank.bank_id, self.input_tile.get_memsize()))
        return traces

class UnicastLayer:
    ''' Unicast Layer.
    Args:
        uid: unique identifier for the layer
        src: source node index
        dst: destination node index
        dims: dimensions of the vector to be unicasted
        cores: list of Core objects available for mapping operations
        banks: list of MemoryBank objects available for mapping tiles
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, src, dst, graph, dims, wafer, prec) -> None:
        self.uid = uid
        self.src = src
        self.dst = dst
        self.dims = dims
        self.wafer = wafer 
        self.prec = prec

        # self.vector_dim = eval("*".join([str(d) for d in dims]))
        # self.tile_size = intceil(self.vector_dim/self.wafer.num_cores_per_node)

        self.tile_size = 16

        self.graph_op = graph.get_op(src, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=dims,
            prec=self.prec,
        )
        self.input_tensor.map_to_memory(wafer.banks[src])

        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=dims,
            prec=self.prec,
        )
        self.output_tensor.map_to_memory(wafer.banks[dst])

        self.in_tiles = {}
        self.out_tiles = {}
        self.tile_ops = {}

        self.create_tiles()
        self.create_ops()

        self.map()

    def create_tiles(self):
        def _create1d(self):
            for d0, pD0 in enumerate(range(0, self.dims[0], self.tile_size)):
                tiled_B = min(self.tile_size, self.dims[0] - pD0)
                # self.in_tiles[b] = Tile("{}_{}".format(self.uid, b), [tiled_B,], prec=self.prec)
                self.in_tiles[d0] = self.input_tensor.slice([(pD0, pD0 + tiled_B),])
                # self.out_tiles[b] = Tile("{}_{}".format(self.uid, b), [tiled_B,], prec=self.prec)
                self.out_tiles[d0] = self.output_tensor.slice([(pD0, pD0 + tiled_B),])

        def _create2d(self):
            pass 

        if len(self.dims) == 1:
            _create1d(self)
        else:
            raise NotImplementedError("UnicastLayer only supports 1D vector unicast for now.")

    def create_ops(self):
        for b in self.in_tiles:
            self.tile_ops[b] = TileUnicastOp("{}_unicast_{}".format(self.uid, b), self.in_tiles[b], self.out_tiles[b])

    def map(self):
        self.map_tiles()
        self.map_ops()

    def map_tiles(self):
        num_banks_per_node = self.wafer.num_banks_per_node
        for b in self.in_tiles:
            self.in_tiles[b].map_to_memory(self.wafer.get_bank(self.src, b % num_banks_per_node))
            self.out_tiles[b].map_to_memory(self.wafer.get_bank(self.dst, b % num_banks_per_node))

    def map_ops(self):
        num_cores_per_node = self.wafer.num_cores_per_node
        for b in self.tile_ops:
            self.tile_ops[b].map_to_core(self.wafer.get_core(self.src, b % num_cores_per_node))