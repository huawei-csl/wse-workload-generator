import logging 

from src.core_level.common.tile import Tile
from src.node_level.common.utils import dtype_to_byte
from src.node_level.common.utils import intceil

def reset_tensor_registry():
    TensorRegistry.reset()

class TensorRegistry:
    _registry = {}

    @classmethod
    def register(cls, tensor):
        assert tensor.uid not in cls._registry, f"Tensor with uid {tensor.uid} is already registered."
        cls._registry[tensor.uid] = tensor

    @classmethod
    def get(cls, uid):
        return cls._registry.get(uid, None)

    @classmethod
    def is_exist(cls, uid):
        return uid in cls._registry

    @classmethod
    def reset(cls):
        cls._registry = {}

'''
Tensor class representing a multi-dimensional array.
'''
class Tensor:
    def __new__(cls, uid, dims, prec, assert_exist = False, *args, **kwargs):
        if TensorRegistry.is_exist(uid):
            existing_tensor = TensorRegistry.get(uid)

            is_dims_match = existing_tensor.dims_match(dims)
            if not is_dims_match:
                is_squeezable = existing_tensor.is_squeezable(dims)
                assert is_squeezable, "Tensor dimensions do not match for uid {} and not squeezable/unsqueezable. existing dims: {} new dims: {}".format(uid, existing_tensor.dims, dims)
            
                existing_tensor.expand_dims(dims)

            assert existing_tensor.dims == dims, "Tensor dimensions do not match for uid {}. existing dims: {} new dims: {}".format(uid, existing_tensor.dims, dims)
            assert existing_tensor.prec == prec, "Tensor precision does not match for uid {}. existing prec: {} new prec: {}".format(uid, existing_tensor.prec, prec)
            return existing_tensor
        
        if assert_exist:
            raise ValueError("assert_exist flag is set but tensor with uid {} does not exist.".format(uid))
        
        return super().__new__(cls)

    def __init__(self, uid, dims, prec) -> None:
        if not hasattr(self, 'uid'):
            self.uid = uid
            
            try:
                node_id = int(uid.split(":")[0])
            except:
                assert False, f"Tensor uid {uid} is not in the correct format. It should be in the format 'node_id:unique_id'."

            self.dims = dims # List of dimensions
            
            self.n_dims = len(dims) # Number of dimensions
            assert self.n_dims in [1, 2, 3, 4], "Only 1D, 2D, 3D, and 4D tensors are supported."

            self.tile_size = None
            self.memory_map = None
            self.addr_offset = None 

            self.prec = prec 
            TensorRegistry.register(self)

    '''
    Slice the tensor into a tile given the indices.
    Args:
        indices: List of tuples representing the start and end indices for each dimension, in the form of [(start1, end1), (start2, end2), ...]
    Returns:
        Tile object representing the sliced tensor.
    '''
    def slice(self, indices):
        assert len(indices) == len(self.dims), "Slice range does not match tensor dimensions."
        
        tile_id = "{}[{}]".format(self.uid, ",".join([f"{start}:{end}" for start, end in indices]))
        tile = Tile(tile_id, self, indices, prec=self.prec)
        
        return tile

    '''
    Map the tensor to memory banks using FRACTAL_Z layout, where each block of size block_size is mapped to a memory bank in a round-robin fashion.
    Example mapping for a 32x48 tensor with block size 16 and 4 memory banks:
    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | 0 | 1 |
    -------------
    Args:
        banks: List of MemoryBank objects to map the tensor to.
        addr_offset: Optional bank offset to start mapping from a specific bank. It should be an integer in the range [0, number of banks - 1].
    Returns:
        memory_map: Dictionary representing the memory mapping of the tensor.
    '''
    def map_to_memory(self, banks, tile_size, addr_offset=None):
        def _map1d(self):
            num_banks = len(banks)
            self.memory_map = {}
            bank_id = self.addr_offset

            for i in range(0, intceil(self.dims[0]/tile_size[0])):
                start_i = i * tile_size[0]
                end_i = min(start_i + tile_size[0], self.dims[0])
                self.memory_map[i] = {"range": [(start_i, end_i),], "bank": banks[bank_id % num_banks]}
                bank_id += 1

        def _map2d(self):
            num_banks = len(banks)
            self.memory_map = {}
            bank_id = self.addr_offset

            for i in range(0, intceil(self.dims[0]/tile_size[0])):
                self.memory_map[i] = {}
                for j in range(0, intceil(self.dims[1]/tile_size[1])):
                    start_i = i * tile_size[0]
                    start_j = j * tile_size[1]
                    end_i = min(start_i + tile_size[0], self.dims[0])
                    end_j = min(start_j + tile_size[1], self.dims[1])
                    self.memory_map[i][j] = {"range": [(start_i, end_i), (start_j, end_j)], "bank": banks[bank_id % num_banks]}
                    bank_id += 1
        
        def _map3d(self):
            num_banks = len(banks)
            self.memory_map = {}
            bank_id = self.addr_offset

            for i in range(0, intceil(self.dims[0]/tile_size[0])):
                self.memory_map[i] = {}
                for j in range(0, intceil(self.dims[1]/tile_size[1])):
                    self.memory_map[i][j] = {}
                    for k in range(0, intceil(self.dims[2]/tile_size[2])):
                        start_i = i * tile_size[0]
                        start_j = j * tile_size[1]
                        start_k = k * tile_size[2]
                        end_i = min(start_i + tile_size[0], self.dims[0])
                        end_j = min(start_j + tile_size[1], self.dims[1])
                        end_k = min(start_k + tile_size[2], self.dims[2])
                        self.memory_map[i][j][k] = {"range": [(start_i, end_i), (start_j, end_j), (start_k, end_k)], "bank": banks[bank_id % num_banks]}
                        bank_id += 1

        def _map4d(self):
            num_banks = len(banks)
            self.memory_map = {}
            bank_id = self.addr_offset

            for i in range(0, intceil(self.dims[0]/tile_size[0])):
                self.memory_map[i] = {}
                for j in range(0, intceil(self.dims[1]/tile_size[1])):
                    self.memory_map[i][j] = {}
                    for k in range(0, intceil(self.dims[2]/tile_size[2])):
                        self.memory_map[i][j][k] = {}
                        for l in range(0, intceil(self.dims[3]/tile_size[3])):
                            start_i = i * tile_size[0]
                            start_j = j * tile_size[1]
                            start_k = k * tile_size[2]
                            start_l = l * tile_size[3]
                            end_i = min(start_i + tile_size[0], self.dims[0])
                            end_j = min(start_j + tile_size[1], self.dims[1])
                            end_k = min(start_k + tile_size[2], self.dims[2])
                            end_l = min(start_l + tile_size[3], self.dims[3])
                            self.memory_map[i][j][k][l] = {"range": [(start_i, end_i), (start_j, end_j), (start_k, end_k), (start_l, end_l)], "bank": banks[bank_id % num_banks]}
                            bank_id += 1
            
        if self.memory_map is not None:
            logging.debug("Tensor {} is already mapped to memory. Skipping mapping...".format(self.uid))
            return
        
        assert self.tile_size is None, "Tensor {} is already mapped to memory with tile size {}.".format(self.uid, self.tile_size)
        tile_size = list(tile_size)
        assert len(tile_size) == self.n_dims, "Tile size dimensions do not match tensor dimensions."
        self.tile_size = tile_size

        if addr_offset is None:
            self.addr_offset = 0
        else:
            assert 0 <= addr_offset < len(banks), "Address offset out of bounds."          
            self.addr_offset = addr_offset

        if self.n_dims == 1:
            _map1d(self)
        elif self.n_dims == 2:
            _map2d(self)
        elif self.n_dims == 3:
            _map3d(self)
        elif self.n_dims == 4:
            _map4d(self)
        else:
            raise NotImplementedError

    def set_map(self, memory_map, tile_size, addr_offset=None):
        assert self.memory_map is None, "Tensor {} is already mapped to memory.".format(self.uid)
        assert self.tile_size is None, "Tensor {} already has tile_size {}".format(self.uid, self.tile_size)
        
        i = 0
        tmp_map = memory_map
        while i < self.n_dims:
            assert len(tmp_map) == intceil(self.dims[i]/tile_size[i]), "Memory map size does not match tensor dimensions and tile size."
            tmp_map = tmp_map[0]
            i += 1

        if addr_offset is None:
            self.addr_offset = 0
        else:
            self.addr_offset = addr_offset

        self.memory_map = memory_map
        self.tile_size = list(tile_size)
        assert len(self.tile_size) == self.n_dims, "Tile size dimensions do not match tensor dimensions."

        


    '''
    Calculate the physical banks and memory sizes for a given index range.
    Args:
        ind_rng: List of tuples representing the start and end indices for each dimension, in the form of [(start1, end1), (start2, end2), ...]
    Returns:
        mem_sizes: Dictionary mapping MemoryBank objects to the size of data in bytes stored in each bank for the given index range.
    '''
    def get_physical_address(self, ind_rng):
        def _get_address_1d(self, ind_rng):
            start_i, end_i = ind_rng[0]
            assert 0 <= start_i < end_i <= self.dims[0], "Index range out of bounds."

            mem_sizes = {}
            for i in range(start_i // self.tile_size[0], intceil(end_i / self.tile_size[0])):
                block_start_i, block_end_i = self.memory_map[i]["range"][0]

                bank = self.memory_map[i]["bank"]
                if bank not in mem_sizes:
                    mem_sizes[bank] = 0

                len_i = min(block_end_i, end_i) - max(block_start_i, start_i)
                m_size = len_i * dtype_to_byte(self.prec)
                mem_sizes[bank] += m_size

            return mem_sizes

        def _get_address_2d(self, ind_rng):
            start_i, end_i = ind_rng[0]
            start_j, end_j = ind_rng[1]

            assert 0 <= start_i < end_i <= self.dims[0], "Index range out of bounds."
            assert 0 <= start_j < end_j <= self.dims[1], "Index range out of bounds."

            mem_sizes = {} # memory size in each bank in bytes
            for i in range(start_i // self.tile_size[0], intceil(end_i/self.tile_size[0])):
                for j in range(start_j // self.tile_size[1], intceil(end_j / self.tile_size[1])):
                    block_start_i, block_end_i = self.memory_map[i][j]["range"][0]
                    block_start_j, block_end_j = self.memory_map[i][j]["range"][1]

                    bank = self.memory_map[i][j]["bank"]
                    if bank not in mem_sizes:
                        mem_sizes[bank] = 0

                    len_i = min(block_end_i, end_i) - max(block_start_i, start_i)
                    len_j = min(block_end_j, end_j) - max(block_start_j, start_j)
                    m_size = len_i * len_j * dtype_to_byte(self.prec) # in bytes
                    mem_sizes[bank] += m_size

            return mem_sizes

        def _get_address_3d(self, ind_rng):
            start_i, end_i = ind_rng[0]
            start_j, end_j = ind_rng[1]
            start_k, end_k = ind_rng[2]

            assert 0 <= start_i < end_i <= self.dims[0], "Index range out of bounds."
            assert 0 <= start_j < end_j <= self.dims[1], "Index range out of bounds."
            assert 0 <= start_k < end_k <= self.dims[2], "Index range out of bounds."
            
            mem_sizes = {}
            for i in range(start_i // self.tile_size[0], intceil(end_i / self.tile_size[0])):
                for j in range(start_j // self.tile_size[1], intceil(end_j / self.tile_size[1])):
                    for k in range(start_k // self.tile_size[2], intceil(end_k / self.tile_size[2])):
                        block_start_i, block_end_i = self.memory_map[i][j][k]["range"][0]
                        block_start_j, block_end_j = self.memory_map[i][j][k]["range"][1]
                        block_start_k, block_end_k = self.memory_map[i][j][k]["range"][2]

                        bank = self.memory_map[i][j][k]["bank"]
                        if bank not in mem_sizes:
                            mem_sizes[bank] = 0

                        len_i = min(block_end_i, end_i) - max(block_start_i, start_i)
                        len_j = min(block_end_j, end_j) - max(block_start_j, start_j)
                        len_k = min(block_end_k, end_k) - max(block_start_k, start_k)
                        m_size = len_i * len_j * len_k * dtype_to_byte(self.prec)
                        mem_sizes[bank] += m_size
            return mem_sizes

        def _get_address_4d(self, ind_rng):
            start_i, end_i = ind_rng[0]
            start_j, end_j = ind_rng[1]
            start_k, end_k = ind_rng[2]
            start_l, end_l = ind_rng[3]

            assert 0 <= start_i < end_i <= self.dims[0], "Index range out of bounds."
            assert 0 <= start_j < end_j <= self.dims[1], "Index range out of bounds."
            assert 0 <= start_k < end_k <= self.dims[2], "Index range out of bounds."
            assert 0 <= start_l < end_l <= self.dims[3], "Index range out of bounds."

            mem_sizes = {}
            for i in range(start_i // self.tile_size[0], intceil(end_i / self.tile_size[0])):
                for j in range(start_j // self.tile_size[1], intceil(end_j / self.tile_size[1])):
                    for k in range(start_k // self.tile_size[2], intceil(end_k / self.tile_size[2])):
                        for l in range(start_l // self.tile_size[3], intceil(end_l / self.tile_size[3])):
                            block_start_i, block_end_i = self.memory_map[i][j][k][l]["range"][0]
                            block_start_j, block_end_j = self.memory_map[i][j][k][l]["range"][1]
                            block_start_k, block_end_k = self.memory_map[i][j][k][l]["range"][2]
                            block_start_l, block_end_l = self.memory_map[i][j][k][l]["range"][3]                                  

                            bank = self.memory_map[i][j][k][l]["bank"]
                            if bank not in mem_sizes:
                                mem_sizes[bank] = 0

                            len_i = min(block_end_i, end_i) - max(block_start_i, start_i)
                            len_j = min(block_end_j, end_j) - max(block_start_j, start_j)
                            len_k = min(block_end_k, end_k) - max(block_start_k, start_k)
                            len_l = min(block_end_l, end_l) - max(block_start_l, start_l)

                            m_size = len_i * len_j * len_k * len_l * dtype_to_byte(self.prec)
                            mem_sizes[bank] += m_size

            return mem_sizes

        assert self.memory_map is not None, "Tensor {} is not mapped to memory.".format(self.uid)
        assert self.tile_size is not None, "Tensor {} does not have tile size.".format(self.uid)
        assert len(ind_rng) == len(self.dims), "Index range does not match tensor dimensions."

        if self.n_dims == 1:
            return _get_address_1d(self, ind_rng) 
        elif self.n_dims == 2:
            return _get_address_2d(self, ind_rng)
        elif self.n_dims == 3:
            return _get_address_3d(self, ind_rng)
        elif self.n_dims == 4:
            return _get_address_4d(self, ind_rng)
        else:
            raise NotImplementedError

    '''
    Get the total memory footprint of the tensor per bank in bytes.
    Args:
        None
    Returns:
        mem_sizes: Dictionary mapping MemoryBank objects to the size of data in bytes stored in each bank for the entire tensor.
    '''
    def get_mem_footprint(self):
        assert self.memory_map is not None, "Tensor {} is not mapped to memory.".format(self.uid)

        # Simply call the get_physical_address function with the full index range
        ind_rng = [(0, d) for d in self.dims]
        return self.get_physical_address(ind_rng)

    def is_squeezable(self, target_dims):
        my_dims = list(self.dims)
        actions = []

        i = 0
        j = 0

        while True:
            if i >= len(my_dims) and j >= len(target_dims):
                return True, actions # both reached the end
            
            elif i >= len(my_dims):
                if target_dims[j] != 1:
                    return False, [] # my dims reached end, but target dims has non-1 dim left
                # my dims reached end, but target dims has 1, so unsqueeze myself once
                actions.append("u")
                j += 1

            elif j >= len(target_dims):
                if my_dims[i] != 1:
                    return False, [] # target reached end, but my dims has non-1 dim left
                # target reached end, but I still have a 1, so squeeze myself once
                actions.append("s") 
                i += 1

            else:
                if my_dims[i] == target_dims[j]:
                    # dims match, do nothing but move on
                    i += 1
                    j += 1
                    actions.append(None)

                elif my_dims[i] == 1:
                    # my dims has a 1, so squeeze myself once
                    actions.append("s")
                    i += 1

                elif target_dims[j] == 1:
                    # target dims has a 1, so unsqueeze myself once
                    actions.append("u")
                    j += 1

                else:
                    # dims do not match, thus not squeezable
                    return False, []

    def dims_match(self, other_dims):
        if len(other_dims) != len(self.dims):
            return False 

        for i in range(len(self.dims)):
            if other_dims[i] != self.dims[i]:
                return False
            
        return True 

    def squeeze(self, dim):
        def _remove_dim_recurse(mmap, curr_dim, target_dim):
            if curr_dim < target_dim:
                for key in mmap:
                    _remove_dim_recurse(mmap[key], curr_dim + 1, target_dim)
            elif curr_dim == target_dim:
                assert len(mmap) == 1, "Cannot squeeze dimension {} with size greater than 1.".format(target_dim)

                key = list(mmap.keys())[0]

                tmp_map = mmap[key]
                for i in mmap[key]:
                    mmap[i] = tmp_map[i]
            else:
                assert False, "Should not reach here."

        def _update_slice_range(mmap, dim):
            if "range" in mmap:
                mmap["range"].pop(dim)
                return
            else:
                for key in mmap:
                    _update_slice_range(mmap[key], dim)

        assert -self.n_dims <= dim < self.n_dims, "Dimension out of bounds."
        if dim < 0:
            dim += self.n_dims

        assert self.dims[dim] == 1, "Cannot squeeze dimension {} with size {}.".format(dim, self.dims[dim])

        self.dims.pop(dim)
        self.n_dims = len(self.dims)

        self.tile_size.pop(dim)

        _remove_dim_recurse(self.memory_map, 0, dim)
        _update_slice_range(self.memory_map, dim)

    def unsqueeze(self, dim):
        def _add_dim_recurse(mmap, curr_dim, target_dim):
            if curr_dim < target_dim:
                for key in mmap:
                    _add_dim_recurse(mmap[key], curr_dim + 1, target_dim)
            elif curr_dim == target_dim:
                tmp_mmap = dict(mmap)
                mmap.clear()
                mmap[0] = tmp_mmap
            else:
                assert False, "Should not reach here."

        def _update_slice_range(mmap, dim):
            if "range" in mmap:
                mmap["range"].insert(dim, (0,1))
                return
            else:
                for key in mmap:
                    _update_slice_range(mmap[key], dim)
            
        assert -self.n_dims <= dim < self.n_dims, "Dimension out of bounds."
        if dim < 0:
            dim += self.n_dims

        self.dims.insert(dim, 1)
        self.n_dims = len(self.dims)

        self.tile_size.insert(dim, 1)
        
        _add_dim_recurse(self.memory_map, 0, dim)
        _update_slice_range(self.memory_map, dim)

    def expand_dims(self, new_dims):
        is_squeezable, actions = self.is_squeezable(new_dims)
        assert is_squeezable, "Cannot expand tensor {} by squeezing/unsqueezing from dims {} to new dims {}.".format(self.uid, self.dims, new_dims)

        d = 0
        for action in actions:
            if action == "s":
                self.squeeze(d)
            elif action == "u":
                self.unsqueeze(d)
                d += 1
            else:
                d += 1

        assert self.dims == new_dims, "Dims do not match after squeeze/unsqueeze. my_dims: {}, target: {}".format(self.dims, new_dims)

    def clone(self, new_uid):
        new_tensor = Tensor(new_uid, list(self.dims), self.prec)
        new_tensor.tile_size = list(self.tile_size) if self.tile_size else None
        new_tensor.memory_map = dict(self.memory_map) if self.memory_map else None
        return new_tensor

if __name__=="__main__":
    from src.core_level.common.wafer import Wafer

    wafer = Wafer([4,4], [6,6])

    reset_tensor_registry()
    
    node_id = 0
    dims = [2, 3]
    tile_size = [1, 1]

    addr_offset = 0

    tensor_a = Tensor(f"{node_id}:A", dims, "fp16")
    tensor_a.map_to_memory(wafer.banks[node_id], tile_size, addr_offset=addr_offset)