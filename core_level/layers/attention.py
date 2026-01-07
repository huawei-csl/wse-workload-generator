import logging
from typing import List

from core_level.common.graph import get_compute_graph
from core_level.common.tensor import Tensor
from core_level.common.tile import load_tiling_config
from core_level.common.wafer import Core
from core_level.layers.reduce import TileReduceOp
from core_level.common.isa import InstructionSet

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
            # traces.append("READ {} {} {}".format(self.q_tile.id, bank.bank_id, size))
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))

        # Read KV tile from memory
        mem_sizes = self.kv_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            # traces.append("READ {} {} {}".format(self.kv_tile.id, bank.bank_id, size))
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))

        for b in range(B):
            # traces.append(f"GEMM {self.q_tile.id}[{b}] {self.kv_tile.id}[{b}] {self.out_tile.id}[{b}] {H}x{D}x{S_kv}")
            traces.append(InstructionSet.GEMM([H, D, S_kv], self.id))

        for b in range(B):
            # traces.append(f"GEMM {self.q_tile.id}[{b}] {self.kv_tile.id}[{b}] {self.out_tile.id}[{b}] {H}x{S_kv}x{Do}")
            traces.append(InstructionSet.GEMM([H, S_kv, Do], self.id))

        # Write output tile back to memory
        mem_sizes = self.out_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            # traces.append("WRITE {} {} {}".format(self.out_tile.id, bank.bank_id, size))
            traces.append(InstructionSet.WRITE(bank.bank_id, size, self.id))

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
        wafer: Wafer object representing the hardware architecture
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, node_id, graph, q_dims, kv_dims, pe_dims, wafer, prec) -> None:
        assert len(q_dims) == 4, "dims should be a tuple of (bsz, seqlen_q, num_heads, kv_lora_rank)"
        assert len(kv_dims) == 3, "dims should be a tuple of (bsz, seqlen_kv, kv_lora_rank)"
        assert len(pe_dims) == 3, "dims should be a tuple of (bsz, seqlen_kv, qk_rope_head_dim)"

        self.uid = uid
        self.node_id = node_id
        self.q_dims = q_dims
        self.kv_dims = kv_dims
        self.pe_dims = pe_dims
        self.wafer = wafer

        self.q_tile_size = load_tiling_config("configs/tiling.json", "AttentionQ", self.q_dims)
        self.kv_tile_size = load_tiling_config("configs/tiling.json", "AttentionKV", self.kv_dims)
        self.prec = prec

        self.graph_op = graph.get_op(node_id, uid)

        B, S_q, H, D = q_dims
        B, S_kv, D = kv_dims
        _, _, D_pe = pe_dims
        Tb, Ts_kv, Td = self.kv_tile_size
        
        self.q_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=[B, S_q, H, D+D_pe],
            prec=self.prec,
        )
        
        self.kv_tensor = Tensor(
            uid=self.uid + "_kv",
            dims=[B, S_kv, D+D_pe],
            prec=self.prec,
        )
        
        # store partial results between QKV and epilogue
        self.qkv_out_tensors = []
        for s, pS in enumerate(range(0, S_kv, Ts_kv)):
            qkv_out_tensor = Tensor(
                    uid=self.uid + "_qkv_out_{}".format(s),
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
        for b in self.q_tiles:
            self.tile_ops[b] = {}
            for h in self.q_tiles[b]:
                self.tile_ops[b][h] = {}
                for s in self.kv_tiles[b]:
                    self.tile_ops[b][h][s] = TileAttnOp("{}_attn_{}_{}_{}".format(self.uid,b,h,s), self.q_tiles[b][h], self.kv_tiles[b][s], self.qkv_out_tiles[b][h][s])

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
                    
        for b in self.tile_ops:
            for h in self.tile_ops[b]:
                mem_sizes = self.reduce_ops[b][h].out_tile.get_physical_address()
                assert len(mem_sizes) == 1, "Out tile is mapped to multiple memory banks."
                core_id = list(mem_sizes.keys())[0].local_id

                self.reduce_ops[b][h].map_to_core(self.wafer.get_core(self.node_id, core_id))
