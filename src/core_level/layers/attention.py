import logging
from typing import List

from src.core_level.common.graph import get_compute_graph
from src.core_level.common.tensor import Tensor
from src.core_level.common.tile import load_tiling_config
from src.core_level.common.wafer import Core
from src.core_level.layers.reduce import TileReduceOp
from src.core_level.common.isa import InstructionSet
from src.core_level.common.stats import Stats
from src.node_level.common.utils import byte_to_str, dtype_to_byte

class TileAttnOp:
    def __init__(self, id, q_tile, kv_tile, out_tile) -> None:
        self.id = id
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.out_tile = out_tile

        assert len(q_tile.dims) == 4, "Query tile must be 4D in TileAttnOp {}.".format(self.id)
        assert len(kv_tile.dims) == 3, "Key/Value tile must be 3D in TileAttnOp {}.".format(self.id)
        assert len(out_tile.dims) == 4, "Output tile must be 4D in TileAttnOp {}.".format(self.id)

        assert q_tile.dims[0] == kv_tile.dims[0], "Batch size mismatch in TileAttnOp {}.".format(self.id)
        assert q_tile.dims[0] == out_tile.dims[0], "Batch size mismatch in TileAttnOp {}.".format(self.id)
        assert q_tile.dims[1] == 1, "Query tile sequence length must be 1 in TileAttnOp {}.".format(self.id)
        assert q_tile.dims[2] == out_tile.dims[2], "Head dimension mismatch in TileAttnOp {}.".format(self.id)
        assert q_tile.dims[3] == kv_tile.dims[2], "Hidden dimension mismatch in TileAttnOp {}.".format(self.id)

        self.mapped_core = None
        self.stats = Stats()
        logging.debug("TileAttnOp {} is created with q tile {}, kv tile {}, out tile {}.".format(self.id, self.q_tile.id, self.kv_tile.id, self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileAttnOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileAttnOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))
    
    def get_traces(self) -> List[str]:
        B, _, H, D = self.q_tile.dims
        _, S_kv, _ = self.kv_tile.dims
        _, _, _, Do = self.out_tile.dims

        traces = []

        # Read query tile from memory
        mem_sizes = self.q_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
            self.stats.add_reads(size)

        # Read KV tile from memory
        mem_sizes = self.kv_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
            self.stats.add_reads(size)

        for b in range(B):
            traces.append(InstructionSet.GEMM([H, D, S_kv], self.id))
            self.stats.add_cube(self.mapped_core.core_id, 2 * H * D * S_kv)

        for b in range(B):
            traces.append(InstructionSet.GEMM([H, S_kv, Do], self.id))
            self.stats.add_cube(self.mapped_core.core_id, 2 * H * S_kv * Do)

        # Write output tile back to memory
        mem_sizes = self.out_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.WRITE(bank.bank_id, size, self.id))
            self.stats.add_writes(size)

        return traces

class MLALayer:
    ''' MLA Attention Layer.
    Args:
        uid: unique identifier for the layer
        node_id: node where the layer is mapped
        graph: compute graph object
        q_dims: dimensions of the query tensor (bsz, seqlen_q, num_heads, kv_lora_rank)
        kv_dims: dimensions of the key/value tensor (bsz, seqlen_kv, kv_lora_rank)
        pe_dims: dimensions of the positional encoding tensor (bsz, seqlen_kv, qk_rope_head_dim)
        q_tile_size: tile size for the query tensor
        kv_tile_size: tile size for the key/value tensor
        wafer: Wafer object representing the hardware architecture
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, node_id, graph, q_dims, kv_dims, pe_dims, q_tile_size, kv_tile_size, wafer, prec) -> None:
        assert len(q_dims) == 4, "dims should be a tuple of (bsz, seqlen_q, num_heads, kv_lora_rank)"
        assert len(kv_dims) == 3, "dims should be a tuple of (bsz, seqlen_kv, kv_lora_rank)"
        assert len(pe_dims) == 3, "dims should be a tuple of (bsz, seqlen_kv, qk_rope_head_dim)"

        self.uid = uid
        self.node_id = node_id
        self.q_dims = q_dims
        self.kv_dims = kv_dims
        self.pe_dims = pe_dims
        self.wafer = wafer

        self.q_tile_size = q_tile_size
        self.kv_tile_size = kv_tile_size
        self.prec = prec

        self.graph_op = graph.get_op(node_id, uid)

        B, S_q, H, D = q_dims
        B, S_kv, D = kv_dims
        _, _, D_pe = pe_dims
        Tb, Ts_kv, Td = self.kv_tile_size
        
        self.split_kv = S_kv > Ts_kv

        self.q_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=[B, S_q, H, D+D_pe],
            prec=self.prec,
        )
        
        self.kv_tensor = Tensor(
            uid=f"{node_id}:{uid}_kv",
            dims=[B, S_kv, D+D_pe],
            prec=self.prec,
        )

        # store partial results between QKV and epilogue
        self.qkv_out_tensors = []
        if self.split_kv:
            for s, pS in enumerate(range(0, S_kv, Ts_kv)):
                qkv_out_tensor = Tensor(
                        uid=f"{self.node_id}:{self.uid}_qkv_out_{s}",
                        dims=[B, 1, H, D],
                        prec=self.prec,
                    )
                self.qkv_out_tensors.append(qkv_out_tensor)

        # outputs after epilogue
        self.reduce_out_tensor = Tensor(
                    uid=self.graph_op["outputs"][0],
                    dims=[B, S_q, H, D],
                    prec=self.prec,
                )
        
        self.q_tiles = {}
        self.kv_tiles = {}
        self.qkv_out_tiles = {}
        self.reduce_out_tiles = {}

        self.tile_ops = {} 
        self.reduce_ops = {}

        core_ids = [self.wafer.get_core(self.node_id, i).core_id for i in range(self.wafer.num_cores_per_node)]
        self.stats = Stats(core_ids)

        self.create_tiles()
        self.create_ops()

        self.map_ops()

    def create_tiles(self):
        B, S_q, H, D = self.q_dims
        Tb, Ts_q, Th, Td = self.q_tile_size
        assert S_q == 1 and Ts_q == 1, "Only support seqlen_q == 1 for decoding."
        assert D == Td, "Does not support tiling in hidden dim dimension."

        B, S_kv, D = self.kv_dims
        Tb, Ts_kv, Td = self.kv_tile_size
        _, _, D_pe = self.pe_dims

        self.q_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Ts_q, Th, D+D_pe], addr_offset=0)
        self.kv_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Ts_kv, D+D_pe], addr_offset=0)
        self.reduce_out_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Ts_q, Th, D], addr_offset=0)
        for s in range(len(self.qkv_out_tensors)):
            self.qkv_out_tensors[s].map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Ts_q, Th, D], addr_offset=0)

        for b, pB in enumerate(range(0, B, Tb)):
            self.q_tiles[b] = {}
            self.qkv_out_tiles[b] = {}
            self.reduce_out_tiles[b] = {}
            for h, pH in enumerate(range(0, H, Th)):
                tiled_B = min(Tb, B - pB)
                tiled_H = min(Th, H - pH)

                self.q_tiles[b][h] = self.q_tensor.slice([(b*Tb, b*Tb + tiled_B), (0, 1), (h*Th, h*Th + tiled_H), (0, D+D_pe)])
                
                if self.split_kv:
                    self.qkv_out_tiles[b][h] = {}
                    for s, pS in enumerate(range(0, S_kv, Ts_kv)):    
                        self.qkv_out_tiles[b][h][s] = self.qkv_out_tensors[s].slice([(b*Tb, b*Tb + tiled_B), (0, 1), (h*Th, h*Th + tiled_H), (0, D)])
                    
                self.reduce_out_tiles[b][h] = self.reduce_out_tensor.slice([(b*Tb, b*Tb + tiled_B), (0, 1), (h*Th, h*Th + tiled_H), (0, D)])

        for b, pB in enumerate(range(0, B, Tb)):
            self.kv_tiles[b] = {}
            for s, pS in enumerate(range(0, S_kv, Ts_kv)):
                tiled_B = min(Tb, B - pB)
                tiled_S = min(Ts_kv, S_kv - pS)

                self.kv_tiles[b][s] = self.kv_tensor.slice([(b*Tb, b*Tb + tiled_B), (s*Ts_kv, s*Ts_kv + tiled_S), (0, D+D_pe)])
    
    def create_ops(self):
        _, S_kv, _ = self.kv_dims
        _, Ts_kv, _ = self.kv_tile_size

        for b in self.q_tiles:
            self.tile_ops[b] = {}
            for h in self.q_tiles[b]:
                self.tile_ops[b][h] = {}
                for s in self.kv_tiles[b]:
                    if self.split_kv:
                        out_tile = self.qkv_out_tiles[b][h][s]  
                    else:
                        assert s == 0, "When not using split-K, there should be only one KV tile."
                        out_tile = self.reduce_out_tiles[b][h]
                    self.tile_ops[b][h][s] = TileAttnOp("{}_attn_{}_{}_{}".format(self.uid,b,h,s), self.q_tiles[b][h], self.kv_tiles[b][s], out_tile)

        if self.split_kv:
            for b in self.q_tiles:
                self.reduce_ops[b] = {}
                for h in self.q_tiles[b]:
                    self.reduce_ops[b][h] = TileReduceOp("{}_attn_reduce_{}_{}".format(self.uid,b,h), 
                                                        [self.tile_ops[b][h][s].out_tile for s in self.tile_ops[b][h]], 
                                                        self.reduce_out_tiles[b][h])

    def map_ops(self):
        for b in self.tile_ops:
            for h in self.tile_ops[b]:
                for s in self.tile_ops[b][h]:
                    mem_sizes = self.tile_ops[b][h][s].kv_tile.get_physical_address()
                    assert len(mem_sizes) == 1, "KV tile is mapped to multiple memory banks."
                    core_id = list(mem_sizes.keys())[0].local_id

                    self.tile_ops[b][h][s].map_to_core(self.wafer.get_core(self.node_id, core_id))
                    self.stats.merge(self.tile_ops[b][h][s].stats)

        for b in self.reduce_ops:
            for h in self.reduce_ops[b]:
                mem_sizes = self.reduce_ops[b][h].out_tile.get_physical_address()
                assert len(mem_sizes) == 1, "Out tile is mapped to multiple memory banks."
                core_id = list(mem_sizes.keys())[0].local_id

                self.reduce_ops[b][h].map_to_core(self.wafer.get_core(self.node_id, core_id))
                self.stats.merge(self.reduce_ops[b][h].stats)

    def calc_expected(self):
        B, S_q, H, D = self.q_dims
        B, S_kv, D = self.kv_dims
        B, S_kv, D_pe = self.pe_dims

        expected = {
            "q_size": B * S_q * H * (D+D_pe) * dtype_to_byte(self.prec),
            "kv_size": B * S_kv * (D+D_pe) * dtype_to_byte(self.prec),
            "output_size": B * S_q * H * D * dtype_to_byte(self.prec),
            "flops": 
                2 * B * S_q * H * S_kv * D # Q x KV
                + 2 * B * S_q * H * S_kv * D_pe # Q x PE
                + 2 * B * S_q * H * S_kv * D, # A x KV
        }
        expected["reads"] = expected["q_size"] + expected["kv_size"]
        expected["writes"] = expected["output_size"]
        return expected
    
    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, expected=expected, dims=[list(self.q_dims), list(self.kv_dims)], tile_size=[self.q_tile_size, self.kv_tile_size])

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
        op_id = f"{node_id}:mla_0"
        ops[node_id][op_id] = {
            "type": "MLAAbsorbAttention",
            "inputs": [f"{node_id}:mla_0_q"],
            "outputs": [f"{node_id}:output_tensor"]
        }

    bsz = 2
    seqlen_q = 1
    seqlen_kv = 8
    num_heads = 4
    kv_lora_rank = 16
    qk_rope_head_dim = 16
    
    q_tile_size = [2, 1, 4, 16]
    kv_tile_size = [2, 8, 16]

    q_dims = bsz, seqlen_q, num_heads, kv_lora_rank
    kv_dims = bsz, seqlen_kv, kv_lora_rank
    pe_dims = bsz, seqlen_kv, qk_rope_head_dim

    q_tensor = Tensor(
        uid=f"{node_id}:mla_0_q",
        dims=[bsz, seqlen_q, num_heads, kv_lora_rank+qk_rope_head_dim],
        prec="fp16",
    )
    
    kv_tensor = Tensor(
        uid=f"{node_id}:mla_0_kv",
        dims=[bsz, seqlen_kv, kv_lora_rank+qk_rope_head_dim],
        prec="fp16",
    )

    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    for node_id in range(wafer.num_nodes):
        layer = MLALayer(f"{node_id}:mla_0", node_id, graph, q_dims, kv_dims, pe_dims, q_tile_size, kv_tile_size, wafer, prec="fp16")
        layer.log_stats()

    traces = wafer.get_traces()
    for node_id in traces:
        for core_id in range(wafer.num_cores_per_node):
            print(core_id)
            for trace in traces[node_id][core_id]:
                parsed_instr = InstructionSet.parse(trace)
                print(parsed_instr)