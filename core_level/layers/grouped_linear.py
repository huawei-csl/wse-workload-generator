import json
import logging
from typing import List

from core_level.common.graph import get_compute_graph
from core_level.common.tensor import Tensor
from core_level.common.tile import load_tiling_config
from core_level.common.wafer import Core
from core_level.layers.reduce import TileReduceOp
from core_level.common.isa import InstructionSet
from core_level.common.stats import Stats
from src.node_level.common.utils import dtype_to_byte

class TileGroupGemmOp:
    def __init__(self, id, input_tile, weight_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.weight_tile = weight_tile
        self.out_tile = out_tile
        self.mapped_core = None
        self.stats = Stats()

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

        # Read input tile from memory
        mem_sizes = self.input_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
            self.stats.add_reads(size)
            # stats["reads"] += size

        # Read weight tile from memory
        mem_sizes = self.weight_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
            self.stats.add_reads(size)
            # stats["reads"] += size

        for b in range(B):
            traces.append(InstructionSet.GEMM([M, K, N], self.id))
            self.stats.add_cube(self.mapped_core.core_id, 2 * M * K * N)
            # stats["flops"] += 2 * M * K * N

        # Write output tile back to memory
        mem_sizes = self.out_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.WRITE(bank.bank_id, size, self.id))
            self.stats.add_writes(size)
            # stats["writes"] += size

        return traces


class GroupedLinearLayer:
    ''' Grouped Linear Layer.
    Args:
        uid: unique identifier for the layer
        node_id: node index where the layer is mapped
        graph: compute graph object
        dims: dimensions of the GroupedGEMM operation (B, M, K, N)
        cores: list of Core objects available for mapping operations
        banks: list of MemoryBank objects available for mapping tiles
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, node_id, graph, dims, wafer, prec) -> None:
        assert len(dims) == 4, "dims should be a tuple of (B, M, K, N)"
        self.uid = uid
        self.node_id = node_id
        self.dims = dims
        self.wafer = wafer

        self.tile_size = load_tiling_config("configs/tiling.json", "GroupedLinear", self.dims)
        self.prec = prec
        
        self.graph_op = graph.get_op(node_id, uid)

        B, M, K, N = dims

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=[B, M, K],
            prec=self.prec,
        )
        
        self.weight_tensor = Tensor(
            uid=self.uid + "_weight",
            dims=[B, K, N],
            prec=self.prec,
        )
        
        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=[B, M, N],
            prec=self.prec,
        )
        
        self.input_tiles = {}
        self.weight_tiles = {}
        self.partial_sum_tiles = {}
        self.out_tiles = {} 
        self.tile_ops = {} 
        self.reduce_ops = {}
        
        core_ids = [self.wafer.get_core(self.node_id, i).core_id for i in range(self.wafer.num_cores_per_node)]
        self.stats = Stats(core_ids)

        self.create_tiles()
        self.create_ops()

        self.map_ops()

    def create_tiles(self):
        """Generate tiling for GroupedGEMM operation."""

        B, M, K, N = self.dims
        Tb, Tm, Tk, Tn = self.tile_size

        self.input_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Tm, Tk], addr_offset=0)
        self.weight_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Tk, Tn], addr_offset=0)
        self.output_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Tm, Tn], addr_offset=0)

        for b, pB in enumerate(range(0, B, Tb)):
            self.input_tiles[b] = {}
            for m, pM in enumerate(range(0, M, Tm)):
                self.input_tiles[b][m] = {}
                for k, pK in enumerate(range(0, K, Tk)):
                    tiled_B = min(Tb, B - pB)
                    tiled_M = min(Tm, M - pM)
                    tiled_K = min(Tk, K - pK)
                    self.input_tiles[b][m][k] = self.input_tensor.slice([(b*Tb, b*Tb + tiled_B), (m*Tm, m*Tm + tiled_M), (k*Tk, k*Tk + tiled_K)])

        for b, pB in enumerate(range(0, B, Tb)):
            self.weight_tiles[b] = {}
            for k, pK in enumerate(range(0, K, Tk)):
                self.weight_tiles[b][k] = {}
                for n, pN in enumerate(range(0, N, Tn)):
                    tiled_B = min(Tb, B - pB)
                    tiled_N = min(Tn, N - pN)
                    tiled_K = min(Tk, K - pK)
                    self.weight_tiles[b][k][n] = self.weight_tensor.slice([(b*Tb, b*Tb + tiled_B), (k*Tk, k*Tk + tiled_K), (n*Tn, n*Tn + tiled_N)])

        # Split-K requires partial sum tiles
        if K != Tk:
            for b, pB in enumerate(range(0, B, Tb)):
                self.partial_sum_tiles[b] = {}
                for m, pM in enumerate(range(0, M, Tm)):
                    self.partial_sum_tiles[b][m] = {}
                    for k, pK in enumerate(range(0, K, Tk)):
                        self.partial_sum_tiles[b][m][k] = {}
                        for n, pN in enumerate(range(0, N, Tn)):
                            tiled_B = min(Tb, B - pB)
                            tiled_M = min(Tm, M - pM)
                            tiled_K = min(Tk, K - pK)
                            tiled_N = min(Tn, N - pN)

                            partial_sum_tensor = Tensor(
                                uid="{}_partial_sum_{}_{}_{}_{}".format(self.uid, b, m, k, n),
                                dims=[tiled_B, tiled_M, tiled_N],
                                prec=self.prec,
                            )

                            mem_sizes = self.weight_tiles[b][k][n].get_physical_address()
                            assert len(mem_sizes) == 1, "Weight tile is mapped to multiple memory banks.".format()
                            core_id = list(mem_sizes.keys())[0].local_id

                            partial_sum_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[tiled_B, tiled_M, tiled_N], addr_offset=core_id)
                            self.partial_sum_tiles[b][m][k][n] = partial_sum_tensor.slice([(0, tiled_B), (0, tiled_M), (0, tiled_N)])

        for b, pB in enumerate(range(0, B, Tb)):
            self.out_tiles[b] = {}
            for m, pM in enumerate(range(0, M, Tm)):
                self.out_tiles[b][m] = {}
                for n, pN in enumerate(range(0, N, Tn)):
                    tiled_B = min(Tb, B - pB)
                    tiled_M = min(Tm, M - pM)
                    tiled_N = min(Tn, N - pN)
                    self.out_tiles[b][m][n] = self.output_tensor.slice([(b*Tb, b*Tb + tiled_B), (m*Tm, m*Tm + tiled_M), (n*Tn, n*Tn + tiled_N)]) 

    def create_ops(self):
        B, M, K, N = self.dims
        Tb, Tm, Tk, Tn = self.tile_size


        for b in self.input_tiles:
            self.tile_ops[b] = {}
            for m in self.input_tiles[b]:
                self.tile_ops[b][m] = {}
                for k in self.input_tiles[b][m]:
                    self.tile_ops[b][m][k] = {}
                    for n in self.weight_tiles[b][k]:
                        if K != Tk:
                            # Split-K requires partial sum tiles
                            self.tile_ops[b][m][k][n] = TileGroupGemmOp("{}_grouped_gemm_{}_{}_{}_{}".format(self.uid,b,m,k,n), self.input_tiles[b][m][k], self.weight_tiles[b][k][n], self.partial_sum_tiles[b][m][k][n])
                        else:
                            self.tile_ops[b][m][k][n] = TileGroupGemmOp("{}_grouped_gemm_{}_{}_{}_{}".format(self.uid,b,m,k,n), self.input_tiles[b][m][k], self.weight_tiles[b][k][n], self.out_tiles[b][m][n])

        if K != Tk:
            # Create reduction ops for partial sums
            for b in self.out_tiles:
                self.reduce_ops[b] = {}
                for m in self.out_tiles[b]:
                    self.reduce_ops[b][m] = {}
                    for n in self.out_tiles[b][m]:
                        self.reduce_ops[b][m][n] = TileReduceOp("{}_reduce_{}_{}_{}".format(self.uid,b,m,n), 
                                                        [self.partial_sum_tiles[b][m][k][n] for k, pK in enumerate(range(0, K, Tk))], 
                                                        self.out_tiles[b][m][n]
                                                    )

    def map_ops(self):
        for b in self.tile_ops:
            for m in self.tile_ops[b]:
                for k in self.tile_ops[b][m]:
                    for n in self.tile_ops[b][m][k]:
                        # core_id = op_cnt % num_cores_per_node
                        mem_sizes = self.tile_ops[b][m][k][n].weight_tile.get_physical_address()
                        assert len(mem_sizes) == 1, "Weight tile is mapped to multiple memory banks.".format()

                        core_id = list(mem_sizes.keys())[0].local_id

                        self.tile_ops[b][m][k][n].map_to_core(self.wafer.get_core(self.node_id, core_id))
                        self.stats.merge(self.tile_ops[b][m][k][n].stats)

        for b in self.reduce_ops:
            for m in self.reduce_ops[b]:
                for n in self.reduce_ops[b][m]:
                    # map reduction op to the core where the output tile is mapped
                    mem_sizes = self.out_tiles[b][m][n].get_physical_address()
                    assert len(mem_sizes) == 1, "Output tile is mapped to multiple memory banks.".format()
                    core_id = list(mem_sizes.keys())[0].local_id

                    self.reduce_ops[b][m][n].map_to_core(self.wafer.get_core(self.node_id, core_id))
                    self.stats.merge(self.reduce_ops[b][m][n].stats)

    def calc_expected(self):
        B, M, K, N = self.dims
        expected = {
            "input0_size": B * M * K * dtype_to_byte(self.prec),
            "input1_size": B * K * N * dtype_to_byte(self.prec),
            "output_size": B * M * N * dtype_to_byte(self.prec),
            "flops": 2 * B * M * K * N
        }
        expected["reads"] = expected["input0_size"] + expected["input1_size"]
        expected["writes"] = expected["output_size"]
        return expected
    

    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, expected=expected, dims=self.dims, tile_size=self.tile_size)
