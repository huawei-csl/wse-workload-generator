import json
import logging
from typing import List

from core_level.common.tile import Tile, load_tiling_config
from core_level.common.wafer import Core

class TileGroupGemmOp:
    def __init__(self, id, input_tile, weight_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.weight_tile = weight_tile
        self.out_tile = out_tile
        self.mapped_core = None

        assert self.input_tile.dims[0] == self.weight_tile.dims[0], "Batch size mismatch in TileGroupGemmOp {}.".format(self.id)
        logging.debug("TileGroupGemmOp {} is created with input tile {}, weight tile {}, out tile {}.".format(self.id, self.input_tile.id, self.weight_tile.id, self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileGroupGemmOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileGroupGemmOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        B, M, K = self.input_tile.dims
        _, _, N = self.weight_tile.dims

        traces = []
        traces.append("READ {} {} {}".format(self.input_tile.id, self.input_tile.mem_bank.bank_id, self.input_tile.get_memsize()))
        traces.append("READ {} {} {}".format(self.weight_tile.id, self.weight_tile.mem_bank.bank_id, self.weight_tile.get_memsize()))

        for b in range(B):
            traces.append(f"GEMM {self.input_tile.id}[{b}] {self.weight_tile.id}[{b}] {self.out_tile.id}[{b}] {M}x{K}x{N}")

        traces.append("WRITE {} {} {}".format(self.out_tile.id, self.out_tile.mem_bank.bank_id, self.out_tile.get_memsize()))
        return traces


class GroupedLinearLayer:
    ''' Grouped Linear Layer.
    Args:
        uid: unique identifier for the layer
        dims: dimensions of the GroupedGEMM operation (B, M, K, N)
        cores: list of Core objects available for mapping operations
        banks: list of MemoryBank objects available for mapping tiles
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, node_id, dims, wafer, prec) -> None:
        assert len(dims) == 4, "dims should be a tuple of (B, M, K, N)"
        self.uid = uid
        self.node_id = node_id
        self.dims = dims
        self.wafer = wafer

        self.tile_size = load_tiling_config("configs/tiling.json", "GroupedLinear", self.dims)
        self.prec = prec

        self.input_tiles = {}
        self.weight_tiles = {}
        self.out_tiles = {} 
        self.tile_ops = {} 

        self.create_tiles()
        self.create_ops()

        self.map()

    def create_tiles(self):
        """Generate tiling for GroupedGEMM operation."""

        B, M, K, N = self.dims
        Tb, Tm, Tk, Tn = self.tile_size

        assert K == Tk, "Split-K is not supported."

        for b, pB in enumerate(range(0, B, Tb)):
            self.input_tiles[b] = {}
            for m, pM in enumerate(range(0, M, Tm)):
                self.input_tiles[b][m] = {}
                for k, pK in enumerate(range(0, K, Tk)):
                    tiled_B = min(Tb, B - pB)
                    tiled_M = min(Tm, M - pM)
                    tiled_K = min(Tk, K - pK)
                    self.input_tiles[b][m][k] = Tile("{}_input_{}_{}_{}".format(self.uid, b,m,k), [tiled_B, tiled_M, tiled_K], prec=self.prec)

        for b, pB in enumerate(range(0, B, Tb)):
            self.weight_tiles[b] = {}
            for k, pK in enumerate(range(0, K, Tk)):
                self.weight_tiles[b][k] = {}
                for n, pN in enumerate(range(0, N, Tn)):
                    tiled_B = min(Tb, B - pB)
                    tiled_N = min(Tn, N - pN)
                    tiled_K = min(Tk, K - pK)
                    self.weight_tiles[b][k][n] = Tile("{}_weight_{}_{}_{}".format(self.uid, b,k,n), [tiled_B, tiled_K, tiled_N], prec=self.prec)

        for b, pB in enumerate(range(0, B, Tb)):
            self.out_tiles[b] = {}
            for m, pM in enumerate(range(0, M, Tm)):
                self.out_tiles[b][m] = {}
                for n, pN in enumerate(range(0, N, Tn)):
                    tiled_B = min(Tb, B - pB)
                    tiled_M = min(Tm, M - pM)
                    tiled_N = min(Tn, N - pN)
                    self.out_tiles[b][m][n] = Tile("{}_out_{}_{}_{}".format(self.uid,b,m,n), [tiled_B, tiled_M, tiled_N], prec=self.prec)

    def create_ops(self):
        k = 0
        for b in self.input_tiles:
            self.tile_ops[b] = {}
            for m in self.input_tiles[b]:
                self.tile_ops[b][m] = {k: {}}
                for n in self.weight_tiles[b][k]:
                    self.tile_ops[b][m][k][n] = TileGroupGemmOp("{}_gemm_{}_{}_{}_{}".format(self.uid,b,m,k,n), self.input_tiles[b][m][k], self.weight_tiles[b][k][n], self.out_tiles[b][m][n])

    def map_tiles(self):
        num_banks_per_node = self.wafer.num_banks_per_node

        k = 0
        cnt = 0
        for b in self.input_tiles:
            for m in self.input_tiles[b]:
                self.input_tiles[b][m][k].map_to_memory(self.wafer.get_bank(self.node_id, cnt % num_banks_per_node))
                cnt += 1
        cnt = 0
        for b in self.weight_tiles:
            for n in self.weight_tiles[b][k]:
                self.weight_tiles[b][k][n].map_to_memory(self.wafer.get_bank(self.node_id, cnt % num_banks_per_node))
                cnt += 1

    def map(self):
        self.map_tiles()
        self.map_ops()

    def map_ops(self):
        num_cores_per_node = self.wafer.num_cores_per_node

        k = 0
        op_cnt = 0
        for b in self.tile_ops:
            for m in self.tile_ops[b]:
                for n in self.tile_ops[b][m][k]:
                    core_id = op_cnt % num_cores_per_node
                    self.tile_ops[b][m][k][n].map_to_core(self.wafer.get_core(self.node_id, core_id))
                    
                    # For data locality, map the output tile to the memory of the same core
                    if not self.tile_ops[b][m][k][n].out_tile.is_mapped():
                        self.tile_ops[b][m][k][n].out_tile.map_to_memory(self.wafer.get_bank(self.node_id, core_id))

                    op_cnt += 1
        