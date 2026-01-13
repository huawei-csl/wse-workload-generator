
from typing import List
import logging 

from core_level.common.isa import InstructionSet
from core_level.common.stats import Stats

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
        for tile in self.in_tiles:
            mem_sizes = tile.get_physical_address()
            for bank, size in mem_sizes.items():
                traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
                self.stats.add_reads(size)
                # stats["reads"] += size

            traces.append(InstructionSet.ADD(self.out_tile.dims, self.id))
            self.stats.add_vector(self.mapped_core.core_id, eval("*".join(map(str, self.out_tile.dims))))

        # Write output tile back to memory
        mem_sizes = self.out_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.WRITE(bank.bank_id, size, self.id))
            self.stats.add_writes(size)
            # stats["writes"] += size

        return traces
    