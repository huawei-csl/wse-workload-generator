import logging
from typing import List

from src.node_level.common.utils import dtype_to_byte

from src.core_level.common.wafer import Core
from src.core_level.layers.reduce import TileReduceOp

from src.core_level.common.tensor import Tensor
from src.core_level.common.isa import InstructionSet
from src.core_level.common.stats import Stats

class TileGemmOp:
    def __init__(self, id, input_tile, weight_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.weight_tile = weight_tile
        self.out_tile = out_tile

        assert self.input_tile.dims[1] == self.weight_tile.dims[0], "TileGemmOp {} has incompatible input tile {} and weight tile {}.".format(self.id, self.input_tile.id, self.weight_tile.id)

        self.mapped_core = None
        self.stats = Stats()
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

        # Read input tile from memory
        mem_sizes = self.input_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
            self.stats.add_reads(size)

        # Read weight tile from memory
        mem_sizes = self.weight_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
            self.stats.add_reads(size)

        # Perform GEMM operation
        traces.append(InstructionSet.GEMM([M, K, N], self.id))
        self.stats.add_cube(self.mapped_core.core_id, 2 * M * K * N)

        # Write output tile back to memory
        mem_sizes = self.out_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.WRITE(bank.bank_id, size, self.id))
            self.stats.add_writes(size)

        return traces

class LinearLayer:
    ''' Linear Layer.
    Args:
        uid: unique identifier for the layer
        node_id: node where the layer is mapped
        graph: compute graph object
        dims: dimensions of the GEMM operation (M, K, N)
        tile_size: tile size for the GEMM operation (Tm, Tk, Tn)
        wafer: Wafer object representing the hardware architecture
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, node_id, graph, dims, tile_size, wafer, prec) -> None:
        assert len(dims) == 3, "dims should be a tuple of (M, K, N)"

        self.uid = uid
        self.node_id = node_id
        self.dims = dims
        self.tile_size = tile_size
        self.wafer = wafer
        self.prec = prec

        self.graph_op = graph.get_op(node_id, uid)
        
        input_dims = [dims[0], dims[1]]
        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=input_dims,
            prec=self.prec,
        )

        weight_dims = [dims[1], dims[2]] 
        self.weight_tensor = Tensor(
            uid=f"{node_id}:{uid}_weight",
            dims=weight_dims,
            prec=self.prec,
        )

        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=[dims[0], dims[2]],
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
        """Generate tiling for GEMM operation."""
        M, K, N = self.dims
        Tm, Tk, Tn = self.tile_size

        self.input_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tm, Tk], addr_offset=0)
        for m, pM in enumerate(range(0, M, Tm)):
            self.input_tiles[m] = {}
            for k, pK in enumerate(range(0, K, Tk)):
                tiled_M = min(Tm, M - pM)
                tiled_K = min(Tk, K - pK)

                self.input_tiles[m][k] = self.input_tensor.slice([(m*Tm, m*Tm + tiled_M), (k*Tk, k*Tk + tiled_K)])

        self.weight_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tk, Tn], addr_offset=0)
        for k, pK in enumerate(range(0, K, Tk)):
            self.weight_tiles[k] = {}
            for n, pN in enumerate(range(0, N, Tn)):
                tiled_N = min(Tn, N - pN)
                tiled_K = min(Tk, K - pK)

                self.weight_tiles[k][n] = self.weight_tensor.slice([(k*Tk, k*Tk + tiled_K), (n*Tn, n*Tn + tiled_N)])

        # Split-K requires partial sum tiles
        if K != Tk:
            for m, pM in enumerate(range(0, M, Tm)):
                self.partial_sum_tiles[m] = {}
                for k, pK in enumerate(range(0, K, Tk)):
                    self.partial_sum_tiles[m][k] = {}
                    for n, pN in enumerate(range(0, N, Tn)):
                        tiled_M = min(Tm, M - pM)
                        tiled_K = min(Tk, K - pK)
                        tiled_N = min(Tn, N - pN)
                        partial_sum_tensor = Tensor(
                            uid=f"{self.node_id}:{self.uid}_partial_sum_{m}_{k}_{n}",
                            dims=[tiled_M, tiled_N],
                            prec=self.prec,
                        )
                        
                        mem_sizes = self.weight_tiles[k][n].get_physical_address()
                        assert len(mem_sizes) == 1, "Weight tile is mapped to multiple memory banks.".format()
                        core_id = list(mem_sizes.keys())[0].local_id

                        partial_sum_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[tiled_M, tiled_N], addr_offset=core_id)
                        self.partial_sum_tiles[m][k][n] = partial_sum_tensor.slice([(0, tiled_M), (0, tiled_N)])

        self.output_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tm, Tn], addr_offset=0)
        for m, pM in enumerate(range(0, M, Tm)):
            self.out_tiles[m] = {}
            for n, pN in enumerate(range(0, N, Tn)):
                tiled_M = min(Tm, M - pM)
                tiled_N = min(Tn, N - pN)
                self.out_tiles[m][n] = self.output_tensor.slice([(m*Tm, m*Tm + tiled_M), (n*Tn, n*Tn + tiled_N)])

    def create_ops(self):
        M, K, N = self.dims
        Tm, Tk, Tn = self.tile_size

        for m in self.input_tiles:
            self.tile_ops[m] = {}
            for k in self.input_tiles[m]:
                self.tile_ops[m][k] = {}
                for n in self.weight_tiles[k]:
                    if K != Tk:
                        # Split-K requires partial sum tiles
                        self.tile_ops[m][k][n] = TileGemmOp("{}_gemm_{}_{}_{}".format(self.uid,m,k,n), self.input_tiles[m][k], self.weight_tiles[k][n], self.partial_sum_tiles[m][k][n])
                    else:
                        self.tile_ops[m][k][n] = TileGemmOp("{}_gemm_{}_{}_{}".format(self.uid,m,k,n), self.input_tiles[m][k], self.weight_tiles[k][n], self.out_tiles[m][n])

        if K != Tk:
            # Create reduction ops for partial sums
            for m in self.out_tiles:
                self.reduce_ops[m] = {}
                for n in self.out_tiles[m]:
                    self.reduce_ops[m][n] = TileReduceOp("{}_reduce_{}_{}".format(self.uid,m,n), 
                                                    [self.partial_sum_tiles[m][k][n] for k, pK in enumerate(range(0, K, Tk))], 
                                                    self.out_tiles[m][n]
                                                )
    
    def map_ops(self):
        for m in self.tile_ops:
            for k in self.tile_ops[m]:
                for n in self.tile_ops[m][k]:
                    # maximize weight locality by mapping the op to the core where the weight tile is mapped
                    mem_sizes = self.tile_ops[m][k][n].weight_tile.get_physical_address()
                    assert len(mem_sizes) == 1, "Weight tile is mapped to multiple memory banks.".format()
                    core_id = list(mem_sizes.keys())[0].local_id

                    self.tile_ops[m][k][n].map_to_core(self.wafer.get_core(self.node_id, core_id))
                    self.stats.merge(self.tile_ops[m][k][n].stats)

        for m in self.reduce_ops:
            for n in self.reduce_ops[m]:
                # map reduction op to the core where the output tile is mapped
                mem_sizes = self.out_tiles[m][n].get_physical_address()
                assert len(mem_sizes) == 1, "Output tile is mapped to multiple memory banks.".format()
                core_id = list(mem_sizes.keys())[0].local_id

                self.reduce_ops[m][n].map_to_core(self.wafer.get_core(self.node_id, core_id))
                self.stats.merge(self.reduce_ops[m][n].stats)

    def calc_expected(self):
        M, K, N = self.dims
        expected = {
            "input0_size": M * K * dtype_to_byte(self.prec),
            "input1_size": K * N * dtype_to_byte(self.prec),
            "output_size": M * N * dtype_to_byte(self.prec),
            "flops": 2 * M * K * N
        }
        expected["reads"] = expected["input0_size"] + expected["input1_size"]
        expected["writes"] = expected["output_size"]
        return expected
    
    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, expected=expected, dims=self.dims, tile_size=self.tile_size)

if __name__=="__main__":
    from src.core_level.common.wafer import Wafer
    from src.core_level.common.tensor import reset_tensor_registry
    from src.core_level.common.graph import Graph

    reset_tensor_registry()

    node_grid = (1, 1)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:linear_0"
        ops[node_id][op_id] = {
            "type": "Linear",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    dims = [8, 4, 2]
    tile_size = [2, 2, 2]

    input_tensor = Tensor(
        uid="0:input_tensor",
        dims=[dims[0], dims[1]],
        prec="fp16",
    )

    for node_id in range(wafer.num_nodes):
        layer = LinearLayer(f"{node_id}:linear_0", node_id, graph, dims, tile_size, wafer, prec="fp16")
        layer.log_stats()
