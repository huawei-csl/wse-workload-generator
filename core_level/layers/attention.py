import logging
from typing import List

from core_level.common.tile import Tile, load_tiling_config
from core_level.common.wafer import Core

class TileAttnOp:
    def __init__(self, id, q_tile, kv_tile, out_tile) -> None:
        self.id = id
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.out_tile = out_tile
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
        traces.append("READ {} {} {}".format(self.q_tile.id, self.q_tile.mem_bank.bank_id, self.q_tile.get_memsize()))
        traces.append("READ {} {} {}".format(self.kv_tile.id, self.kv_tile.mem_bank.bank_id, self.kv_tile.get_memsize()))

        for b in range(B):
            traces.append(f"GEMM {self.q_tile.id}[{b}] {self.kv_tile.id}[{b}] {self.out_tile.id}[{b}] {H}x{D}x{S_kv}")
        
        for b in range(B):
            traces.append(f"GEMM {self.q_tile.id}[{b}] {self.kv_tile.id}[{b}] {self.out_tile.id}[{b}] {H}x{S_kv}x{Do}")

        traces.append("WRITE {} {} {}".format(self.out_tile.id, self.out_tile.mem_bank.bank_id, self.out_tile.get_memsize()))
        return traces

class TileReduceOp:
    def __init__(self, id, in_tiles, out_tile) -> None:
        self.id = id
        self.in_tiles = in_tiles
        for tile in self.in_tiles[1:]:
            assert tile.dims == in_tiles[0].dims, "Input tiles must have the same dimensions in TileReduceOp {}.".format(self.id)

        self.out_tile = out_tile
        assert out_tile.dims == in_tiles[0].dims, "Output tile must have the same dimensions as input tiles in TileReduceOp {}.".format(self.id)

        self.mapped_core = None
        logging.debug("TileReduceOp {} is created with in_tiles {}, out tile {}.".format(self.id, [t.id for t in self.in_tiles], self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileReduceOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileReduceOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        traces = []
        for tile in self.in_tiles:
            traces.append("READ {} {} {}".format(tile.id, tile.mem_bank.bank_id, tile.get_memsize()))
        traces.append("REDUCE {} {}".format(self.out_tile.id, self.out_tile.dims))
        traces.append("WRITE {} {} {}".format(self.out_tile.id, self.out_tile.mem_bank.bank_id, self.out_tile.get_memsize()))
        return traces
    
class MLALayer:
    ''' MLA Attention Layer.
    Args:
        uid: unique identifier for the layer
        node_id: node where the layer is mapped
        q_dims: dimensions of the query tensor (bsz, seqlen_q, num_heads, kv_lora_rank)
        kv_dims: dimensions of the key/value tensor (bsz, seqlen_kv, kv_lora_rank)
        pe_dims: dimensions of the positional encoding tensor (bsz, seqlen_kv, qk_rope_head_dim)
        wafer: Wafer object representing the hardware architecture
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, node_id, q_dims, kv_dims, pe_dims, wafer, prec) -> None:
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

        self.q_tiles = {}
        self.kv_tiles = {}
        self.qkv_out_tiles = {}
        self.reduce_out_tiles = {}

        self.tile_ops = {} 
        self.reduce_ops = {}

        self.create_tiles()
        self.create_ops()

        self.map()

    def create_tiles(self):
        B, S_q, H, D = self.q_dims
        Tb, Ts_q, Th, Td = self.q_tile_size
        assert S_q == 1 and Ts_q == 1, "Only support seqlen_q == 1 for decoding."
        assert D == Td, "Does not support tiling in hidden dim dimension."

        B, S_kv, D = self.kv_dims
        Tb, Ts_kv, Td = self.kv_tile_size
        _, _, D_pe = self.pe_dims

        for b, pB in enumerate(range(0, B, Tb)):
            self.q_tiles[b] = {}
            self.qkv_out_tiles[b] = {}
            self.reduce_out_tiles[b] = {}
            for h, pH in enumerate(range(0, H, Th)):
                tiled_B = min(Tb, B - pB)
                tiled_H = min(Th, H - pH)
                self.q_tiles[b][h] = Tile("{}_q_{}_{}".format(self.uid, b, h), [tiled_B, 1, tiled_H, D+D_pe], prec=self.prec)
                self.qkv_out_tiles[b][h] = {}
                for s, pS in enumerate(range(0, S_kv, Ts_kv)):
                    self.qkv_out_tiles[b][h][s] = Tile("{}_qkv_out_{}_{}_{}".format(self.uid, b, h, s), [tiled_B, 1, tiled_H, D], prec=self.prec)
                self.reduce_out_tiles[b][h] = Tile("{}_reduce_out_{}_{}".format(self.uid, b, h), [tiled_B, 1, tiled_H, D], prec=self.prec)

        for b, pB in enumerate(range(0, B, Tb)):
            self.kv_tiles[b] = {}
            for s, pS in enumerate(range(0, S_kv, Ts_kv)):
                tiled_B = min(Tb, B - pB)
                tiled_S = min(Ts_kv, S_kv - pS)
                self.kv_tiles[b][s] = Tile("{}_kv_{}_{}".format(self.uid, b, s), [tiled_B, tiled_S, D+D_pe], prec=self.prec)

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


    def map(self):
        self.map_tiles()
        self.map_ops()

    def map_tiles(self):
        num_banks_per_node = self.wafer.num_banks_per_node

        cnt = 0
        for b in self.q_tiles:
            for h in self.q_tiles[b]:
                self.q_tiles[b][h].map_to_memory(self.wafer.get_bank(self.node_id, cnt % num_banks_per_node))
                cnt += 1

        cnt = 0
        for b in self.kv_tiles:
            for s in self.kv_tiles[b]:
                self.kv_tiles[b][s].map_to_memory(self.wafer.get_bank(self.node_id, cnt % num_banks_per_node))
                cnt += 1

    def map_ops(self):
        num_cores = self.wafer.num_cores_per_node

        op_cnt = 0
        for b in self.tile_ops:
            for h in self.tile_ops[b]:
                for s in self.tile_ops[b][h]:
                    core_id = op_cnt % num_cores
                    self.tile_ops[b][h][s].map_to_core(self.wafer.get_core(self.node_id, core_id))
                    
                    # For data locality, map the output tile to the memory of the same core
                    if not self.tile_ops[b][h][s].out_tile.is_mapped():
                        self.tile_ops[b][h][s].out_tile.map_to_memory(self.wafer.get_bank(self.node_id, core_id))

                # map all s tiles for (b,h) to the same core to avoid moving partial results around
                op_cnt += 1

        op_cnt = 0
        for b in self.tile_ops:
            for h in self.tile_ops[b]:
                core_id = op_cnt % num_cores
                self.reduce_ops[b][h].map_to_core(self.wafer.get_core(self.node_id, core_id))

                if not self.reduce_ops[b][h].out_tile.is_mapped():
                    self.reduce_ops[b][h].out_tile.map_to_memory(self.wafer.get_bank(self.node_id, core_id))

                op_cnt += 1