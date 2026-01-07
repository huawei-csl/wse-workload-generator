import os
import logging
from typing import List


def write_to_csv(dat, fname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    with open(fname, "w") as f:
        for line in dat:
            f.write(line)
            f.write("\n")

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

class Wafer:
    def __init__(self, node_grid, core_grid) -> None:
        assert len(node_grid) == 2, "node_grid should be a tuple of (rows, columns)"
        self.node_grid = node_grid  # (rows, columns)
        self.num_nodes = node_grid[0] * node_grid[1]

        self.core_grid = core_grid  # (rows, columns)
        self.num_cores_per_node = core_grid[0] * core_grid[1]
        self.num_banks_per_node= core_grid[0] * core_grid[1]  # assuming same grid for cores and memory banks

        self.banks = {}
        self.cores = {}
        for node_id in range(self.num_nodes):
            self.banks[node_id] = {}
            self.cores[node_id] = {}
            for i in range(self.num_banks_per_node):
                self.banks[node_id][i] = MemoryBank(node_id, i, self.num_banks_per_node)

            for i in range(self.num_cores_per_node):
                self.cores[node_id][i] = Core(node_id, i, self.num_cores_per_node)

    def get_bank(self, node_id: int, local_id: int) -> MemoryBank:
        return self.banks[node_id][local_id]
    
    def get_core(self, node_id: int, local_id: int) -> Core:
        return self.cores[node_id][local_id]
    
    def get_traces(self):
        traces = {}
        for node_id in range(self.num_nodes):
            traces[node_id] = {}
            for core_id in range(self.num_cores_per_node):
                traces[node_id][core_id] = self.cores[node_id][core_id].generate_traces()
        return traces

    def export_traces(self, iter:int, dir_path: str):
        traces = self.get_traces()
        for node_id in range(self.num_nodes):
            for core_id in range(self.num_cores_per_node):

                for trace in traces[node_id][core_id]:
                    logging.debug("Core {}: {}".format(core_id, trace))

                out_fname = dir_path + "/node_{}/{}/core_{}.csv".format(node_id, iter, core_id)
                write_to_csv(traces[node_id][core_id], out_fname)

