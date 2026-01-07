
import pytest
from core_level.common.tensor import Tensor, reset_tensor_registry

@pytest.mark.parametrize(
        "tile_size,addr_offset,node_id,dims,ind_rng,expected",
        [
            # A simple test case where the slice falls onto the first bank only
            (
                [16, 16], 0, 0, [32, 32], [(10, 12), (4, 8)], 
                {
                    "bank_ids": [0],
                    "sizes": [16]
                }
            ),
            # A test case where the slice spans multiple banks in both dimensions
            (
                [16, 16], 0, 0, [32, 32], [(10, 20), (20, 23)], 
                {
                    "bank_ids": [1, 3], # bank_id is iterated first over dim1 then dim0 (inner loop iterates over dim1). 
                    "sizes": [36, 24]
                }
            ),
            # A test case where tensor dimensions are not multiples of block size
            (
                [16, 16], 0, 0, [32, 40], [(10, 20), (36, 40)], 
                {
                    "bank_ids": [2, 5], # bank_id is iterated first over dim1 then dim0 (inner loop iterates over dim1). 
                    "sizes": [48, 32]
                }
            ),
            # A test case with addr_offset > 0
            (
                [16, 16], 2, 0, [32, 32], [(10, 20), (20, 23)],
                {
                    "bank_ids": [2+1, 2+3], # bank_id is iterated first over dim1 then dim0 (inner loop iterates over dim1). 
                    "sizes": [36, 24]
                }
            ),
            # A test case with node_id > 0
            (
                [16, 16], 0, 2, [32, 32], [(10, 20), (20, 23)], 
                {
                    "bank_ids": [72+1, 72+3], # bank_id is iterated first over dim1 then dim0 (inner loop iterates over dim1). 
                    "sizes": [36, 24]
                }
            ),
            # A test case with a non-square tile size
            (
                [8, 16], 0, 0, [32, 32], [(6, 10), (20, 23)], 
                {
                    "bank_ids": [1, 3], # bank_id is iterated first over dim1 then dim0 (inner loop iterates over dim1). 
                    "sizes": [12, 12]
                }
            ),
        ],
)
def test_tensor_memory_layout(tile_size, addr_offset, node_id, dims, ind_rng, expected):
    from core_level.common.wafer import Wafer

    wafer = Wafer([4,4], [6,6])

    reset_tensor_registry()
    
    tensor_a = Tensor("A", dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=addr_offset)

    mem_sizes = tensor_a.get_physical_address(ind_rng)

    num_slices = len(expected["bank_ids"])
    for i in range(num_slices):
        bank = list(mem_sizes.keys())[i]    
        assert expected["bank_ids"][i] == bank.bank_id, "Expected bank id {}, got {}".format(expected["bank_ids"][i], bank.bank_id)
        
        size = mem_sizes[bank]
        assert expected["sizes"][i] == size, "Expected size {}, got {}".format(expected["sizes"][i], size)


if __name__=="__main__":
    test_tensor_memory_layout(
        [8, 16], 0, 0, [32, 32], [(6, 10), (20, 23)], 
        {
            "bank_ids": [1, 3], # bank_id is iterated first over dim1 then dim0 (inner loop iterates over dim1). 
            "sizes": [12, 12]
        }
    )