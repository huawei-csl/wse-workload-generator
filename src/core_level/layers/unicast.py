import logging

from typing import List

from src.node_level.common.utils import dtype_to_byte
from src.core_level.common.stats import Stats
from src.core_level.common.isa import InstructionSet
from src.core_level.common.tensor import Tensor

class TileUnicastOp:
    def __init__(self, id, input_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.out_tile = out_tile
        self.mapped_core = None
        self.stats = Stats()
        logging.debug("TileUnicastOp {} is created with input tile {}, out tile {}.".format(self.id, self.input_tile.id, self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileUnicastOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileUnicastOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        traces = []

        send_mem_sizes = self.input_tile.get_physical_address()
        recv_mem_sizes = self.out_tile.get_physical_address()
        assert len(send_mem_sizes) == len(recv_mem_sizes), "Mismatched number of memory banks between send0 and next tile in TileUnicastOp {}.".format(self.id)

        for i in range(len(send_mem_sizes)):
            send_bank = list(send_mem_sizes.keys())[i]
            send_size = send_mem_sizes[send_bank]

            recv_bank = list(recv_mem_sizes.keys())[i]
            recv_size = recv_mem_sizes[recv_bank]

            assert send_size == recv_size, "Mismatched send0 and next tile sizes in TileUnicastOp {}.".format(self.id)
            
            traces.append(InstructionSet.COPY(send_bank.bank_id, recv_bank.bank_id, send_size, self.id))
            self.stats.add_copy(send_size)

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
    def __init__(self, uid, node_id, src, dst, graph, dims, wafer, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.src = src
        self.dst = dst
        self.dims = dims
        self.wafer = wafer 
        self.prec = prec

        self.graph_op = graph.get_op(src, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=dims,
            prec=self.prec,
        )
        assert self.input_tensor.tile_size is not None, "Input tensor {} of Multicast operation {} on node {} does not have tile size.".format(self.input_tensor.uid, uid, src)
        self.tile_size = list(self.input_tensor.tile_size)

        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=dims,
            prec=self.prec,
        )

        self.in_tiles = {}
        self.out_tiles = {}
        self.tile_ops = {}

        self.stats = Stats()

        self.create_tiles()
        self.create_ops()

        self.map_ops()

    def create_tiles(self):
        # self.output_tensor.map_to_memory(self.wafer.banks[self.dst], tile_size=self.tile_size, addr_offset=0)
        self.output_tensor.set_map(self.input_tensor.memory_map, self.tile_size)

        # Tile from last dimension
        for d0, pD0 in enumerate(range(0, self.dims[-1], self.tile_size[-1])):
            tiled_B = min(self.tile_size[-1], self.dims[-1] - pD0)

            slice_indices = [(0, d) for d in self.dims[0:-1]] + [(pD0, pD0 + tiled_B)]

            self.in_tiles[d0] = self.input_tensor.slice(slice_indices)
            self.out_tiles[d0] = self.output_tensor.slice(slice_indices)

    def create_ops(self):
        for b in self.in_tiles:
            self.tile_ops[b] = TileUnicastOp("{}_unicast_{}".format(self.uid, b), self.in_tiles[b], self.out_tiles[b])

    def map_ops(self):
        dedicated_core = self.wafer.get_core(self.src, 0)
        for b in self.tile_ops:
            self.tile_ops[b].map_to_core(dedicated_core)
            self.stats.merge(self.tile_ops[b].stats)

    def calc_expected(self):
        expected = {"copy": eval("*".join(map(str, self.dims))) * dtype_to_byte(self.input_tensor.prec)}
        expected["reads"] = 0
        expected["writes"] = 0
        return expected

    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.src, expected=expected, dims=self.dims, tile_size=self.tile_size)
