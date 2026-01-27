import json
import logging 
from typing import List
from src.node_level.common.utils import dtype_to_byte

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

def load_tiling_config(filepath: str, layer_type: str, dims: List[int], uid: str = None) -> List[int]:
    def _fill_missing(tile_size, dims):
        for d in range(len(tile_size)):
            # If tile size is None, use the full dimension size
            if tile_size[d] is None:
                tile_size[d] = dims[d]
        return tile_size
    
    def _compare_with_wildcard(str1, str2):
        assert "*" not in str1, "Wildcard '*' is only supported in the second string."
        assert str2.count("*") <= 1, "Only one wildcard '*' is supported in the second string."

        parts = str2.split("*")
        return str1.startswith(parts[0]) and str1.endswith(parts[1])

    '''
    Return the matching key from uids that matches the given key. Supports wildcard '*'.
    '''
    def _key_match(key, uids):
        if key in uids:
            return key
        for uid in uids:
            if _compare_with_wildcard(key, uid):
                return uid
        return None

    with open(filepath, "r") as f:
        tilings = json.load(f)

    if uid:
        matched_key = _key_match(uid, list(tilings["layers"].keys()))
        if matched_key:
            tile_size = tilings["layers"][matched_key]
            assert len(tile_size) == len(dims), "Tiling dimensions do not match GEMM dimensions."
            tile_size = _fill_missing(tile_size, dims)
            return tile_size
    
    tile_size = tilings["defaults"][layer_type]
    assert len(tile_size) == len(dims), "Tiling dimensions do not match GEMM dimensions."
    tile_size = _fill_missing(tile_size, dims)

    return tile_size
    