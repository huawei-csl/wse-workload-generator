import csv
import os 
import shutil

from utils import dtype_to_byte

import logging
from logger import init_logger
from typing import List 

class Tile:
    def __init__(self, id, dims, prec) -> None:
        self.id = id
        self.dims = dims
        self.mem_bank = None
        self.prec = prec # in str, e.g., "fp16", "fp8"
        logging.debug("Tile {} is created with dims {}.".format(self.id, self.dims, self.prec))

    def map_to_memory(self, mem_bank: "MemoryBank"):
        assert self.mem_bank is None, "Tile {} is already mapped to memory {}.".format(self.id, self.mem_bank.bank_id)
        self.mem_bank = mem_bank
        mem_bank.alloc_tile(self)
        logging.debug("Tile {} is mapped to memory {}.".format(self.id, self.mem_bank.bank_id))

    def is_mapped(self):
        return self.mem_bank is not None

    def get_memsize(self):
        memsize = 1 
        for d in self.dims:
            memsize *= d
        return memsize * dtype_to_byte(self.prec)

class TileGemmOp:
    def __init__(self, id, input_tile, weight_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.weight_tile = weight_tile
        self.out_tile = out_tile
        self.mapped_core = None

        logging.debug("TileGemmOp {} is created with input tile {}, weight tile {}, out tile {}.".format(self.id, self.input_tile.id, self.weight_tile.id, self.out_tile.id))

    def map_to_core(self, core: "Core"):
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

class TileMulticastOp:
    def __init__(self, id, input_tile, out_tiles) -> None:
        self.id = id
        self.input_tile = input_tile
        self.out_tiles = out_tiles
        self.mapped_core = None
        logging.debug("TileMulticastOp {} is created with input tile {}, out tiles {}.".format(self.id, self.input_tile.id, [t.id for t in self.out_tiles]))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileMulticastOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileMulticastOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        traces = []
        for i in range(len(self.out_tiles)):
            traces.append("COPY {} {} {} {} {}".format(self.input_tile.id, self.input_tile.mem_bank.bank_id, self.out_tiles[i].id, self.out_tiles[i].mem_bank.bank_id, self.input_tile.get_memsize()))
        return traces

class TileUnicastOp:
    def __init__(self, id, input_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.out_tile = out_tile
        self.mapped_core = None
        logging.debug("TileUnicastOp {} is created with input tile {}, out tile {}.".format(self.id, self.input_tile.id, self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileUnicastOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileUnicastOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        traces = []
        traces.append("COPY {} {} {} {} {}".format(self.input_tile.id, self.input_tile.mem_bank.bank_id, self.out_tile.id, self.out_tile.mem_bank.bank_id, self.input_tile.get_memsize()))
        return traces

class Core:
    def __init__(self, node_id: int, local_id: int, num_cores_per_node: int) -> None:
        self.node_id = node_id
        self.local_id = local_id
        self.core_id = node_id * num_cores_per_node + local_id
        self.instruction_queue = []

    def add_instruction(self, tile_op: "TileGemmOp"):
        self.instruction_queue.append(tile_op)
        logging.debug("TileGemmOp {} is added to core {} instruction queue.".format(tile_op.id, self.core_id))

    def generate_traces(self):
        traces = []
        for op in self.instruction_queue:
            traces += op.get_traces()
        return traces
    
class MemoryBank:
    def __init__(self, node_id: int, local_id: int, num_banks_per_node: int) -> None:
        self.node_id = node_id
        self.local_id = local_id
        self.bank_id = node_id * num_banks_per_node + local_id
        self.allocated_tiles = []

    def alloc_tile(self, tile: "Tile"):
        self.allocated_tiles.append(tile)
        logging.debug("Tile {} is allocated to memory bank {}.".format(tile.id, self.bank_id))

def tile_gemm(layer_id, dims, tile_sizes):
    """Generate tiling for GEMM operation.
    Args:
        dims (tuple): A tuple of three integers (M, K, N) representing the dimensions of the matrices.
        tile_sizes (tuple): A tuple of three integers (Tm, Tk, Tn) representing the tile sizes for M, K, and N dimensions.
    Returns:
        input_tiles (dict): A nested dictionary of input tiles indexed by (m, k).
        weight_tiles (dict): A nested dictionary of weight tiles indexed by (k, n).
        out_tiles (dict): A nested dictionary of output tiles indexed by (m, n).
    """

    M, K, N = dims
    Tm, Tk, Tn = tile_sizes

    assert K == Tk, "Split-K is not supported."

    input_tiles = {}
    for m, pM in enumerate(range(0, M, Tm)):
        input_tiles[m] = {}
        for k, pK in enumerate(range(0, K, Tk)):
            tiled_M = min(Tm, M - pM)
            tiled_K = min(Tk, K - pK)
            input_tiles[m][k] = Tile("{}_input_{}_{}".format(layer_id, m,k), [tiled_M, tiled_K], prec=prec)

    weight_tiles = {}
    for k, pK in enumerate(range(0, K, Tk)):
        weight_tiles[k] = {}
        for n, pN in enumerate(range(0, N, Tn)):
            tiled_N = min(Tn, N - pN)
            tiled_K = min(Tk, K - pK)
            weight_tiles[k][n] = Tile("{}_weight_{}_{}".format(layer_id, k,n), [tiled_K, tiled_N], prec=prec)

    out_tiles = {}
    for m, pM in enumerate(range(0, M, Tm)):
        out_tiles[m] = {}
        for n, pN in enumerate(range(0, N, Tn)):
            tiled_M = min(Tm, M - pM)
            tiled_N = min(Tn, N - pN)
            out_tiles[m][n] = Tile("{}_out_{}_{}".format(layer_id,m,n), [tiled_M, tiled_N], prec=prec)

    return input_tiles, weight_tiles, out_tiles

def tile_group_gemm(layer_id, dims, tile_sizes):
    """Generate tiling for GroupedGEMM operation.
    Args:
        dims (tuple): A tuple of three integers (B, M, K, N) representing the dimensions of the matrices.
        tile_sizes (tuple): A tuple of three integers (Tb, Tm, Tk, Tn) representing the tile sizes for B, M, K, and N dimensions.
    Returns:
        input_tiles (dict): A nested dictionary of input tiles indexed by (b, m, k).
        weight_tiles (dict): A nested dictionary of weight tiles indexed by (b, k, n).
        out_tiles (dict): A nested dictionary of output tiles indexed by (b, m, n).
    """

    B, M, K, N = dims
    Tb, Tm, Tk, Tn = tile_sizes

    assert K == Tk, "Split-K is not supported."

    input_tiles = {}
    for b, pB in enumerate(range(0, B, Tb)):
        input_tiles[b] = {}
        for m, pM in enumerate(range(0, M, Tm)):
            input_tiles[b][m] = {}
            for k, pK in enumerate(range(0, K, Tk)):
                tiled_B = min(Tb, B - pB)
                tiled_M = min(Tm, M - pM)
                tiled_K = min(Tk, K - pK)
                input_tiles[b][m][k] = Tile("{}_input_{}_{}_{}".format(layer_id, b,m,k), [tiled_B, tiled_M, tiled_K], prec=prec)

    weight_tiles = {}
    for b, pB in enumerate(range(0, B, Tb)):
        weight_tiles[b] = {}
        for k, pK in enumerate(range(0, K, Tk)):
            weight_tiles[b][k] = {}
            for n, pN in enumerate(range(0, N, Tn)):
                tiled_B = min(Tb, B - pB)
                tiled_N = min(Tn, N - pN)
                tiled_K = min(Tk, K - pK)
                weight_tiles[b][k][n] = Tile("{}_weight_{}_{}_{}".format(layer_id, b,k,n), [tiled_B, tiled_K, tiled_N], prec=prec)

    out_tiles = {}
    for b, pB in enumerate(range(0, B, Tb)):
        out_tiles[b] = {}
        for m, pM in enumerate(range(0, M, Tm)):
            out_tiles[b][m] = {}
            for n, pN in enumerate(range(0, N, Tn)):
                tiled_B = min(Tb, B - pB)
                tiled_M = min(Tm, M - pM)
                tiled_N = min(Tn, N - pN)
                out_tiles[b][m][n] = Tile("{}_out_{}_{}_{}".format(layer_id,b,m,n), [tiled_B, tiled_M, tiled_N], prec=prec)

    return input_tiles, weight_tiles, out_tiles

def tile_attention(layer_id, q_dims, kv_dims, pe_dims, q_tile_size, kv_tile_size):
    B, S_q, H, D = q_dims
    Tb, Ts_q, Th, Td = q_tile_size
    assert S_q == 1 and Ts_q == 1, "Only support seqlen_q == 1 for decoding."
    assert D == Td, "Does not support tiling in hidden dim dimension."

    B, S_kv, D = kv_dims
    Tb, Ts_kv, Td = kv_tile_size
    _, _, D_pe = pe_dims

    q_tiles = {}
    out_tiles = {}
    for b, pB in enumerate(range(0, B, Tb)):
        q_tiles[b] = {}
        out_tiles[b] = {}
        for h, pH in enumerate(range(0, H, Th)):
            tiled_B = min(Tb, B - pB)
            tiled_H = min(Th, H - pH)
            q_tiles[b][h] = Tile("{}_q_{}_{}".format(layer_id, b, h), [tiled_B, 1, tiled_H, D+D_pe], prec=prec)
            out_tiles[b][h] = Tile("{}_out_{}_{}".format(layer_id, b, h), [tiled_B, 1, tiled_H, D], prec=prec)

    kv_tiles = {}
    for b, pB in enumerate(range(0, B, Tb)):
        kv_tiles[b] = {}
        for s, pS in enumerate(range(0, S_kv, Ts_kv)):
            tiled_B = min(Tb, B - pB)
            tiled_S = min(Ts_kv, S_kv - pS)
            kv_tiles[b][s] = Tile("{}_kv_{}_{}".format(layer_id, b, s), [tiled_B, tiled_S, D+D_pe], prec=prec)

    return q_tiles, kv_tiles, out_tiles

def map_gemm_tiles(input_tiles, weight_tiles, banks):
    num_banks = len(banks)

    k = 0
    for m in input_tiles:
        input_tiles[m][k].map_to_memory(banks[m % num_banks])

    for n in weight_tiles[k]:
        weight_tiles[k][n].map_to_memory(banks[n % num_banks])

def map_group_gemm_tiles(input_tiles, weight_tiles, banks):
    num_banks = len(banks)

    k = 0
    cnt = 0
    for b in input_tiles:
        for m in input_tiles[b]:
            input_tiles[b][m][k].map_to_memory(banks[cnt % num_banks])
            cnt += 1

    cnt = 0
    for b in weight_tiles:
        for n in weight_tiles[b][k]:
            weight_tiles[b][k][n].map_to_memory(banks[cnt % num_banks])
            cnt += 1

def map_attn_tiles(q_tiles, kv_tiles, banks):
    num_banks = len(banks)

    cnt = 0
    for b in q_tiles:
        for h in q_tiles[b]:
            q_tiles[b][h].map_to_memory(banks[cnt % num_banks])
            cnt += 1

    cnt = 0
    for b in kv_tiles:
        for s in kv_tiles[b]:
            kv_tiles[b][s].map_to_memory(banks[cnt % num_banks])
            cnt += 1

def map_tilegemmops(tile_ops, cores, banks):
    num_cores = len(cores)

    k = 0
    op_cnt = 0
    for m in tile_ops:
        for n in tile_ops[m][k]:
            core_id = op_cnt % num_cores
            tile_ops[m][k][n].map_to_core(cores[core_id])
            
            # For data locality, map the output tile to the memory of the same core
            if not tile_ops[m][k][n].out_tile.is_mapped():
                tile_ops[m][k][n].out_tile.map_to_memory(banks[core_id])

            op_cnt += 1

def map_tilegroupgemmops(tile_ops, cores, banks):
    num_cores = len(cores)

    k = 0
    op_cnt = 0
    for b in tile_ops:
        for m in tile_ops[b]:
            for n in tile_ops[b][m][k]:
                core_id = op_cnt % num_cores
                tile_ops[b][m][k][n].map_to_core(cores[core_id])
                
                # For data locality, map the output tile to the memory of the same core
                if not tile_ops[b][m][k][n].out_tile.is_mapped():
                    tile_ops[b][m][k][n].out_tile.map_to_memory(banks[core_id])

                op_cnt += 1

def map_tileattnops(tile_ops, cores, banks):
    num_cores = len(cores)

    op_cnt = 0
    for b in tile_ops:
        for h in tile_ops[b]:
            for s in tile_ops[b][h]:
                core_id = op_cnt % num_cores
                tile_ops[b][h][s].map_to_core(cores[core_id])
                
                # For data locality, map the output tile to the memory of the same core
                if not tile_ops[b][h][s].out_tile.is_mapped():
                    tile_ops[b][h][s].out_tile.map_to_memory(banks[core_id])

            # map all s tiles for (b,h) to the same core to avoid moving partial results around
            op_cnt += 1

def write_to_csv(dat, fname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    with open(fname, "w") as f:
        for line in dat:
            f.write(line)
            f.write("\n")


if __name__=="__main__":
    init_logger(level="INFO")

    curr_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(curr_directory, "traces")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)    

    layers = ["decode5"]
    prec = "fp8"

    decode_iter = 0

    num_nodes = 8
    num_cores_per_node = 24
    num_mem_banks_per_node = num_cores_per_node
    assert num_cores_per_node == num_mem_banks_per_node, "Does not support num_cores != num_mem_banks yet."

    banks = {}
    cores = {}
    for node_id in range(num_nodes):
        banks[node_id] = {}
        cores[node_id] = {}
        for i in range(num_mem_banks_per_node):
            banks[node_id][i] = MemoryBank(node_id, i, num_mem_banks_per_node)

        for i in range(num_cores_per_node):
            cores[node_id][i] = Core(node_id, i, num_cores_per_node)

    for node_id in range(num_nodes):
        logging.info("Generating traces for node {}...".format(node_id))

        csv_fname = "out/decode/node_{}/decode{}.csv".format(node_id, decode_iter)

        # Load CSV
        data = []
        with open(csv_fname, mode="r") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=";")  # Automatically uses the first row as headers
            for row in csv_reader:
                data.append(row)

        # Print loaded data for debugging
        for row in data:
            if row["uid"].split("_")[0] not in layers:
                continue

            if row["operation"] == "Linear":
                continue
                layer_id = row["uid"]

                in_dims = row["Dimensions"].split(" x ")[0][1:-1].split(", ")
                M, K = int(in_dims[0]), int(in_dims[1])
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                assert int(out_dims[0]) == M, "M dimension mismatch."
                M, N = int(out_dims[0]), int(out_dims[1])

                dims = (M, K, N)
                tile_sizes = (16, K, 16)

                input_tiles, weight_tiles, out_tiles = tile_gemm(layer_id, dims, tile_sizes)

                tile_ops = {}

                k = 0
                for m in input_tiles:
                    tile_ops[m] = {k: {}}
                    for n in weight_tiles[k]:
                        tile_ops[m][k][n] = TileGemmOp("{}_gemm_{}_{}_{}".format(layer_id,m,k,n), input_tiles[m][k], weight_tiles[k][n], out_tiles[m][n])

                map_gemm_tiles(input_tiles, weight_tiles, banks[node_id])
                map_tilegemmops(tile_ops, cores[node_id], banks[node_id])

            elif row["operation"] == "GroupedLinear":
                continue
                layer_id = row["uid"]

                in_dims = row["Dimensions"].split(" x ")[0][1:-1].split(", ")
                B, M, K = int(in_dims[0]), int(in_dims[1]), int(in_dims[2])
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                assert int(out_dims[0]) == B, "Batch dimension mismatch."
                assert int(out_dims[1]) == M, "M dimension mismatch."
                B, M, N = int(out_dims[0]), int(out_dims[1]), int(out_dims[2])

                dims = (B, M, K, N)
                tile_sizes = (1, 16, K, 16)

                input_tiles, weight_tiles, out_tiles = tile_group_gemm(layer_id, dims, tile_sizes)

                tile_ops = {}

                k = 0
                for b in input_tiles:
                    tile_ops[b] = {}
                    for m in input_tiles[b]:
                        tile_ops[b][m] = {k: {}}
                        for n in weight_tiles[b][k]:
                            tile_ops[b][m][k][n] = TileGroupGemmOp("{}_gemm_{}_{}_{}_{}".format(layer_id,b,m,k,n), input_tiles[b][m][k], weight_tiles[b][k][n], out_tiles[b][m][n])
                
                map_group_gemm_tiles(input_tiles, weight_tiles, banks[node_id])
                map_tilegroupgemmops(tile_ops, cores[node_id], banks[node_id])

            elif row["operation"] == "MLAAbsorbAttention":
                continue
                layer_id = row["uid"]
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                bsz, seqlen_q, num_heads, kv_lora_rank = int(out_dims[0]), int(out_dims[1]), int(out_dims[2]), int(out_dims[3])
                assert seqlen_q == 1, "Only support seqlen_q == 1 for decoding."
                pe_dims = row["Dimensions"].split(" -> ")[0].split(", PE: ")[-1][1:-1].split(", ")
                assert int(pe_dims[0]) == bsz
                _, seqlen_kv, qk_rope_head_dim = int(pe_dims[0]), int(pe_dims[1]), int(pe_dims[2])

                q_dims = bsz, seqlen_q, num_heads, kv_lora_rank
                kv_dims = bsz, seqlen_kv, kv_lora_rank
                pe_dims = bsz, seqlen_kv, qk_rope_head_dim

                q_tile_size = [4, 1, 16, kv_lora_rank]
                kv_tile_size = [4, 128, kv_lora_rank]

                q_tiles, kv_tiles, out_tiles = tile_attention(layer_id, q_dims, kv_dims, pe_dims, q_tile_size, kv_tile_size)

                tile_ops = {}
                for b in q_tiles:
                    tile_ops[b] = {}
                    for h in q_tiles[b]:
                        tile_ops[b][h] = {}
                        for s in kv_tiles[b]:
                            tile_ops[b][h][s] = TileAttnOp("{}_attn_{}_{}_{}".format(layer_id,b,h,s), q_tiles[b][h], kv_tiles[b][s], out_tiles[b][h])

                map_attn_tiles(q_tiles, kv_tiles, banks[node_id])
                map_tileattnops(tile_ops, cores[node_id], banks[node_id])

            elif row["operation"] == "Multicast":
                layer_id = row["uid"]
                vector_dim = row["Dimensions"][1:-1].split(",")
                assert len(vector_dim) == 1
                vector_dim = int(vector_dim[0])

                comm_group = row["comm. group"][1:-1].split(",")
                dst_nodes = list(map(int, comm_group))

                tile_size = 128

                in_tiles = {}
                out_tiles = {}
                for b, pB in enumerate(range(0, vector_dim, tile_size)):
                    tiled_B = min(tile_size, vector_dim - pB)
                    in_tiles[b] = Tile("{}_{}".format(layer_id, b), [tiled_B,], prec=prec)
                    out_tiles[b] = []
                    for d, dst_node in enumerate(dst_nodes):
                        out_tiles[b].append(Tile("{}_{}_{}".format(layer_id, b, dst_node), [tiled_B,], prec=prec))

                tile_ops = {}
                for b in in_tiles:
                    tile_ops[b] = TileMulticastOp("{}_multicast_{}".format(layer_id, b), in_tiles[b], out_tiles[b])

                # Map tiles
                for b in in_tiles:
                    in_tiles[b].map_to_memory(banks[node_id][b % num_mem_banks_per_node])
                    tile_ops[b].map_to_core(cores[node_id][b % num_cores_per_node])

                    for d, dst_node in enumerate(dst_nodes):
                        out_tiles[b][d].map_to_memory(banks[dst_node][b % num_mem_banks_per_node])

            elif row["operation"] == "Unicast":
                layer_id = row["uid"]
                vector_dim = row["Dimensions"][1:-1].split(",")
                assert len(vector_dim) == 1
                vector_dim = int(vector_dim[0])                

                dst_node = int(row["comm. group"])

                tile_size = 128

                in_tiles = {}
                out_tiles = {}
                for b, pB in enumerate(range(0, vector_dim, tile_size)):
                    tiled_B = min(tile_size, vector_dim - pB)
                    in_tiles[b] = Tile("{}_{}".format(layer_id, b), [tiled_B,], prec=prec)
                    out_tiles[b] = Tile("{}_{}".format(layer_id, b), [tiled_B,], prec=prec)

                tile_ops = {}
                for b in in_tiles:
                    tile_ops[b] = TileUnicastOp("{}_unicast_{}".format(layer_id, b), in_tiles[b], out_tiles[b])

                # Map tiles
                for b in in_tiles:
                    in_tiles[b].map_to_memory(banks[node_id][b % num_mem_banks_per_node])
                    tile_ops[b].map_to_core(cores[node_id][b % num_cores_per_node])
                    out_tiles[b].map_to_memory(banks[dst_node][b % num_mem_banks_per_node])

            else:
                continue 

        for core_id in range(num_cores_per_node):
            traces = cores[node_id][core_id].generate_traces()

            for trace in traces:
                logging.debug("Core {}: {}".format(core_id, trace))

            out_fname = "traces/decode/node_{}/decode{}/core_{}.csv".format(node_id, decode_iter, core_id)
            write_to_csv(traces, out_fname)