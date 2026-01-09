import json
import logging 
from typing import List
from utils import dtype_to_byte

class Tile:
    def __init__(self, id, parent, indices, prec) -> None:
        self.id = id
        self.parent = parent # Parent tensor
        self.indices = indices # indices in the form of [(start1, end1), (start2, end2), ...]
        self.dims = [end - start for start, end in indices]
        self.prec = prec # in str, e.g., "fp16", "fp8"
        logging.debug("Tile {} is created with dims {}.".format(self.id, self.dims, self.prec))

    '''    
    Get the physical address of the tile in the parent tensor's memory.
    Returns:
        mem_sizes: Dictionary mapping MemoryBank objects to the size of data in bytes stored in each bank for the given index range.
    '''
    def get_physical_address(self):
        return self.parent.get_physical_address(self.indices)

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
    