import json
import logging 
from typing import List
from utils import dtype_to_byte

class Tile:
    def __init__(self, id, dims, prec) -> None:
        self.id = id
        self.dims = dims
        self.mem_bank = None
        self.prec = prec # in str, e.g., "fp16", "fp8"
        logging.debug("Tile {} is created with dims {}.".format(self.id, self.dims, self.prec))

    def map_to_memory(self, mem_bank: "MemoryBank"):
        assert self.mem_bank is None, "Tile {} is already mapped to memory {}.".format(self.id, self.mem_bank.bank_id)
        self.mem_bank = mem_bank
        mem_bank.alloc_tile(self)
        logging.debug("Tile {} is mapped to memory {}.".format(self.id, self.mem_bank.bank_id))

    def is_mapped(self):
        return self.mem_bank is not None

    def get_memsize(self):
        memsize = 1 
        for d in self.dims:
            memsize *= d
        return memsize * dtype_to_byte(self.prec)


def load_tiling_config(filepath: str, layer_type: str, dims: List[int]) -> List[int]:
    with open(filepath, "r") as f:
        tilings = json.load(f)
    tile_size = tilings[layer_type]
    assert len(tile_size) == len(dims), "Tiling dimensions do not match GEMM dimensions."
    for d in range(len(tile_size)):
        # If tile size is None, use the full dimension size
        if tile_size[d] is None:
            tile_size[d] = dims[d]
    return tile_size
    