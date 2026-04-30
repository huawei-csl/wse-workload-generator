import logging
from typing import List

from src.core_level.common.wafer import Core
from src.core_level.common.tensor import Tensor
from src.core_level.common.stats import Stats
from src.core_level.common.isa import InstructionSet
from src.core_level.layers.reduce import TileReduceOp
from src.core_level.layers.barrier import Barrier

class TileAttnOp:
    def __init__(self, id, q_tile, kv_tile, out_tile) -> None:
        self.id = id
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.out_tile = out_tile

        assert len(q_tile.dims) == 4, "Query tile must be 4D in TileAttnOp {}.".format(self.id)
        assert len(kv_tile.dims) == 4, "Key/Value tile must be 4D in TileAttnOp {}.".format(self.id)
        assert len(out_tile.dims) == 4, "Output tile must be 4D in TileAttnOp {}.".format(self.id)

        assert q_tile.dims[0] == kv_tile.dims[0], "Batch size mismatch in TileAttnOp {}.".format(self.id)
        assert q_tile.dims[0] == out_tile.dims[0], "Batch size mismatch in TileAttnOp {}.".format(self.id)
        assert q_tile.dims[1] == 1, "Query tile sequence length must be 1 in TileAttnOp {}.".format(self.id)
        assert q_tile.dims[2] == out_tile.dims[2], "Head dimension mismatch in TileAttnOp {}.".format(self.id)
        assert q_tile.dims[3] == kv_tile.dims[3], "Hidden dimension mismatch in TileAttnOp {}.".format(self.id)

        self.mapped_core = None
        self.stats = Stats()
        logging.debug("TileAttnOp {} is created with q tile {}, kv tile {}, out tile {}.".format(self.id, self.q_tile.id, self.kv_tile.id, self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileAttnOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        logging.debug("TileAttnOp {} is mapped to core {}.".format(self.id, core.core_id))
        core.add_instruction(self)
        logging.debug("TileAttnOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        B, _, H, D = self.q_tile.dims
        _, S_kv, _, _ = self.kv_tile.dims
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




class MHALayer:
    def __init__(self, uid, node_id, graph, q_dims, kv_dims, q_tile_size, kv_tile_size, wafer, prec) -> None:
        assert len(q_dims) == 4, "dims should be a tuple of (bsz, seqlen_q, num_heads, head_dim)"
        assert len(kv_dims) == 4, "dims should be a tuple of (bsz, seqlen_kv, num_kv_heads, head_dim)"

        self.uid = uid
        self.node_id = node_id
        self.q_dims = q_dims
        self.kv_dims = kv_dims
        self.wafer = wafer

        self.q_tile_size = q_tile_size
        self.kv_tile_size = kv_tile_size
        self.prec = prec

        self.graph_op = graph.get_op(node_id, uid)

        B, S_q, H_q, D = q_dims
        B, S_kv, H_kv, D = kv_dims

        self.q_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=[B, S_q, H_q, D],
            prec=self.prec,
        )
        
        self.kv_tensor = Tensor(
            uid=f"{node_id}:{uid}_kv",
            dims=[B, S_kv, H_kv, D],
            prec=self.prec,
        )

        Tb, Ts_kv, Th_kv, Td = self.kv_tile_size
        
        self.split_kv = S_kv > Ts_kv

        # store partial results between QKV and epilogue
        self.qkv_out_tensors = []
        if self.split_kv:
            for s, pS in enumerate(range(0, S_kv, Ts_kv)):
                qkv_out_tensor = Tensor(
                        uid=f"{self.node_id}:{self.uid}_qkv_out_{s}",
                        dims=[B, 1, H_q, D], #TODO: Is hardcoded 1 correct here in case of spec dec?
                        prec=self.prec,
                    )
                self.qkv_out_tensors.append(qkv_out_tensor)

        # outputs after epilogue
        self.reduce_out_tensor = Tensor(
                    uid=self.graph_op["outputs"][0],
                    dims=[B, S_q, H_q, D],
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
        B, S_q, H_q, D = self.q_dims
        Tb, Ts_q, Th_q, Td = self.q_tile_size
        assert S_q == 1 and Ts_q == 1, "Only support seqlen_q == 1 for decoding."
        assert D == Td, "Does not support tiling in hidden dim dimension."

        B, S_kv, H_kv, D = self.kv_dims
        Tb, Ts_kv, Th_kv, Td = self.kv_tile_size

        self.q_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Ts_q, Th_q, D], addr_offset=0)
        self.kv_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Ts_kv, Th_kv, D], addr_offset=0)
        self.reduce_out_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Ts_q, Th_q, D], addr_offset=0)
        for s in range(len(self.qkv_out_tensors)):
            self.qkv_out_tensors[s].map_to_memory(self.wafer.banks[self.node_id], tile_size=[Tb, Ts_q, Th_q, D], addr_offset=0)

        for b, pB in enumerate(range(0, B, Tb)):
            self.q_tiles[b] = {}
            self.qkv_out_tiles[b] = {}
            self.reduce_out_tiles[b] = {}
            for h, pH in enumerate(range(0, H_q, Th_q)):
                tiled_B = min(Tb, B - pB)
                tiled_H = min(Th_q, H_q - pH)

                self.q_tiles[b][h] = self.q_tensor.slice([(b*Tb, b*Tb + tiled_B), (0, 1), (h*Th_q, h*Th_q + tiled_H), (0, D)])
                
                if self.split_kv:
                    self.qkv_out_tiles[b][h] = {}
                    for s, pS in enumerate(range(0, S_kv, Ts_kv)):    
                        self.qkv_out_tiles[b][h][s] = self.qkv_out_tensors[s].slice([(b*Tb, b*Tb + tiled_B), (0, 1), (h*Th_q, h*Th_q + tiled_H), (0, D)])
                    
                self.reduce_out_tiles[b][h] = self.reduce_out_tensor.slice([(b*Tb, b*Tb + tiled_B), (0, 1), (h*Th_q, h*Th_q + tiled_H), (0, D)])

        for b, pB in enumerate(range(0, B, Tb)):
            self.kv_tiles[b] = {}
            for h, pH in enumerate(range(0, H_kv, Th_kv)):
                self.kv_tiles[b][h] = {}
                for s, pS in enumerate(range(0, S_kv, Ts_kv)):
                    tiled_B = min(Tb, B - pB)
                    tiled_S = min(Ts_kv, S_kv - pS)
                    tiled_H = min(Th_kv, H_kv - pH)

                    self.kv_tiles[b][h][s] = self.kv_tensor.slice([(b*Tb, b*Tb + tiled_B), (s*Ts_kv, s*Ts_kv + tiled_S),  (h*Th_kv, h*Th_kv + tiled_H), (0, D)])

    
    def create_ops(self):
        for b in self.q_tiles:
            self.tile_ops[b] = {}
            for h in self.q_tiles[b]:
                self.tile_ops[b][h] = {}
                for s in self.kv_tiles[b]:
                    if self.split_kv:
                        out_tile = self.qkv_out_tiles[b][h][s] 
                    else:
                        assert s == 0, "If not splitting kv, there should only be one tile in the sequence dimension."
                        out_tile = self.reduce_out_tiles[b][h]
                    self.tile_ops[b][h][s] = TileAttnOp(
                        f"{self.node_id}:{self.uid}_tile_op_{b}_{h}_{s}",
                        self.q_tiles[b][h],
                        self.kv_tiles[b][h][s],
                        out_tile,
                    )

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

        # Insert barriers between attention ops and reduce phases for split-KV synchronization
        for b in self.reduce_ops:
            for h in self.reduce_ops[b]:
                cores_in_group = []
                for s in self.tile_ops[b][h]:
                    c = self.tile_ops[b][h][s].mapped_core
                    if c not in cores_in_group:
                        cores_in_group.append(c)
                reduce_mem = self.reduce_out_tiles[b][h].get_physical_address()
                reduce_core_id = list(reduce_mem.keys())[0].local_id
                reduce_core = self.wafer.get_core(self.node_id, reduce_core_id)
                if reduce_core not in cores_in_group:
                    cores_in_group.append(reduce_core)
                for core in cores_in_group:
                    Barrier(f"{self.uid}_barrier_reduce_{b}_{h}",
                            cores_in_group).map_to_core(core)

        for b in self.reduce_ops:
            for h in self.reduce_ops[b]:
                mem_sizes = self.reduce_ops[b][h].out_tile.get_physical_address()
                assert len(mem_sizes) == 1, "Out tile is mapped to multiple memory banks."
                core_id = list(mem_sizes.keys())[0].local_id

                self.reduce_ops[b][h].map_to_core(self.wafer.get_core(self.node_id, core_id))
                self.stats.merge(self.reduce_ops[b][h].stats)



    def log_stats(self):
        expected = None
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, expected=expected, dims=[list(self.q_dims), list(self.kv_dims)], tile_size=[self.q_tile_size, self.kv_tile_size])
