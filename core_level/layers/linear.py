import json
import logging
from typing import List

from core_level.common.tile import Tile, load_tiling_config
from core_level.common.wafer import Core

class TileGemmOp:
    def __init__(self, id, input_tile, weight_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.weight_tile = weight_tile
        self.out_tile = out_tile
        self.mapped_core = None

        logging.debug("TileGemmOp {} is created with input tile {}, weight tile {}, out tile {}.".format(self.id, self.input_tile.id, self.weight_tile.id, self.out_tile.id))

    def map_to_core(self, core: Core):
        assert self.mapped_core is None, "TileGemmOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileGemmOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        M, K = self.input_tile.dims
        _, N = self.weight_tile.dims

        traces = []
        traces.append("READ {} {} {}".format(self.input_tile.id, self.input_tile.mem_bank.bank_id, self.input_tile.get_memsize()))
        traces.append("READ {} {} {}".format(self.weight_tile.id, self.weight_tile.mem_bank.bank_id, self.weight_tile.get_memsize()))
        traces.append("GEMM {} {} {} {}".format(self.input_tile.id, self.weight_tile.id, self.out_tile.id, "{}x{}x{}".format(M, K, N)))
        traces.append("WRITE {} {} {}".format(self.out_tile.id, self.out_tile.mem_bank.bank_id, self.out_tile.get_memsize()))
        return traces


class LinearLayer:
    ''' Linear Layer.
    Args:
        uid: unique identifier for the layer
        node_id: node where the layer is mapped
        dims: dimensions of the GEMM operation (M, K, N)
        wafer: Wafer object representing the hardware architecture
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, node_id, dims, wafer, prec) -> None:
        assert len(dims) == 3, "dims should be a tuple of (M, K, N)"

        self.uid = uid
        self.node_id = node_id
        self.dims = dims
        self.wafer = wafer
        self.prec = prec

        self.tile_size = load_tiling_config("configs/tiling.json", "Linear", self.dims)

        self.input_tiles = {}
        self.weight_tiles = {}
        self.out_tiles = {} 
        self.tile_ops = {} 

        self.create_tiles()
        self.create_ops()

        self.map()

    def create_tiles(self):
        """Generate tiling for GEMM operation."""

        M, K, N = self.dims
        Tm, Tk, Tn = self.tile_size

        assert K == Tk, "Split-K is not supported."

        for m, pM in enumerate(range(0, M, Tm)):
            self.input_tiles[m] = {}
            for k, pK in enumerate(range(0, K, Tk)):
                tiled_M = min(Tm, M - pM)
                tiled_K = min(Tk, K - pK)
                self.input_tiles[m][k] = Tile("{}_input_{}_{}".format(self.uid, m,k), [tiled_M, tiled_K], prec=self.prec)

        for k, pK in enumerate(range(0, K, Tk)):
            self.weight_tiles[k] = {}
            for n, pN in enumerate(range(0, N, Tn)):
                tiled_N = min(Tn, N - pN)
                tiled_K = min(Tk, K - pK)
                self.weight_tiles[k][n] = Tile("{}_weight_{}_{}".format(self.uid, k,n), [tiled_K, tiled_N], prec=self.prec)

        for m, pM in enumerate(range(0, M, Tm)):
            self.out_tiles[m] = {}
            for n, pN in enumerate(range(0, N, Tn)):
                tiled_M = min(Tm, M - pM)
                tiled_N = min(Tn, N - pN)
                self.out_tiles[m][n] = Tile("{}_out_{}_{}".format(self.uid,m,n), [tiled_M, tiled_N], prec=self.prec)

    def create_ops(self):
        k = 0
        for m in self.input_tiles:
            self.tile_ops[m] = {k: {}}
            for n in self.weight_tiles[k]:
                self.tile_ops[m][k][n] = TileGemmOp("{}_gemm_{}_{}_{}".format(self.uid,m,k,n), self.input_tiles[m][k], self.weight_tiles[k][n], self.out_tiles[m][n])
    
    def map(self):
        self.map_tiles()
        self.map_ops()

    def map_tiles(self):
        num_banks_per_node = self.wafer.num_banks_per_node

        k = 0
        for m in self.input_tiles:
            self.input_tiles[m][k].map_to_memory(self.wafer.get_bank(self.node_id, m % num_banks_per_node))

        for n in self.weight_tiles[k]:
            self.weight_tiles[k][n].map_to_memory(self.wafer.get_bank(self.node_id, n % num_banks_per_node))

    def map_ops(self):
        num_cores_per_node = self.wafer.num_cores_per_node

        k = 0
        op_cnt = 0
        for m in self.tile_ops:
            for n in self.tile_ops[m][k]:
                # core_id = op_cnt % num_cores_per_node
                # maximize weight locality by mapping the op to the core where the weight tile is mapped
                core_id = self.tile_ops[m][k][n].weight_tile.mem_bank.local_id

                self.tile_ops[m][k][n].map_to_core(self.wafer.get_core(self.node_id, core_id))

                # For data locality, map the output tile to the memory of the same core
                if not self.tile_ops[m][k][n].out_tile.is_mapped():
                    self.tile_ops[m][k][n].out_tile.map_to_memory(self.wafer.get_bank(self.node_id, core_id))

                op_cnt += 1
