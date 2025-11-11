
import logging

from typing import List
from utils import intceil
from core_level.common.tile import Tile

class TileMulticastOp:
    def __init__(self, id, input_tile, out_tiles) -> None:
        self.id = id
        self.input_tile = input_tile
        self.out_tiles = out_tiles
        self.mapped_core = None
        logging.debug("TileMulticastOp {} is created with input tile {}, out tiles {}.".format(self.id, self.input_tile.id, [t.id for t in self.out_tiles]))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileMulticastOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileMulticastOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        traces = []
        for i in range(len(self.out_tiles)):
            traces.append("COPY {} {} {} {} {}".format(self.input_tile.id, self.input_tile.mem_bank.bank_id, self.out_tiles[i].id, self.out_tiles[i].mem_bank.bank_id, self.input_tile.get_memsize()))
        return traces
    
class MulticastLayer:
    ''' Multicast Layer.
    Args:
        uid: unique identifier for the layer
        src: source node index
        dsts: list of destination node indices
        dims: dimensions of the vector to be multicasted
        cores: list of Core objects available for mapping operations
        banks: list of MemoryBank objects available for mapping tiles
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, src, dsts, dims, wafer, prec) -> None:
        self.uid = uid
        self.src = src
        self.dsts = dsts 
        self.wafer = wafer 

        self.vector_dim = eval("*".join([str(d) for d in dims]))
        self.tile_size = intceil(self.vector_dim/self.wafer.num_cores_per_node)

        self.prec = prec

        self.in_tiles = {}
        self.out_tiles = {}
        self.tile_ops = {}

        self.create_tiles()
        self.create_ops()

        self.map()

    def create_tiles(self):
        for b, pB in enumerate(range(0, self.vector_dim, self.tile_size)):
            tiled_B = min(self.tile_size, self.vector_dim - pB)
            self.in_tiles[b] = Tile("{}_{}".format(self.uid, b), [tiled_B,], prec=self.prec)
            self.out_tiles[b] = []
            for d, dst_node in enumerate(self.dsts):
                self.out_tiles[b].append(Tile("{}_{}_{}".format(self.uid, b, dst_node), [tiled_B,], prec=self.prec))

    def create_ops(self):
        for b in self.in_tiles:
            self.tile_ops[b] = TileMulticastOp("{}_multicast_{}".format(self.uid, b), self.in_tiles[b], self.out_tiles[b])

    def map(self):
        self.map_tiles()
        self.map_ops()

    def map_tiles(self):
        num_banks_per_node = self.wafer.num_banks_per_node
        for b in self.in_tiles:
            self.in_tiles[b].map_to_memory(self.wafer.get_bank(self.src, b % num_banks_per_node))
            for d, dst_node in enumerate(self.dsts):
                self.out_tiles[b][d].map_to_memory(self.wafer.get_bank(dst_node, b % num_banks_per_node))

    def map_ops(self):
        num_cores_per_node = self.wafer.num_cores_per_node
        for b in self.in_tiles:
            self.tile_ops[b].map_to_core(self.wafer.get_core(self.src, b % num_cores_per_node))
