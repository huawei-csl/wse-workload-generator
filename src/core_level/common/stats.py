import logging 
import numpy as np

from src.node_level.common.utils import byte_to_str, flops_to_str

class Stats:
    def __init__(self, core_ids=None):
        self.data = {
            "reads": 0,
            "writes": 0,
            "copy": 0,
            "multicast": 0,
            "cube_flops": {core_id: 0 for core_id in core_ids} if core_ids is not None else {},
            "vector_flops": {core_id: 0 for core_id in core_ids} if core_ids is not None else {},
        }
    
    def add_reads(self, size):
        self.data["reads"] += size

    def get_reads(self):
        return self.data["reads"]

    def add_writes(self, size):
        self.data["writes"] += size

    def get_writes(self):
        return self.data["writes"]

    def add_copy(self, size):
        self.data["copy"] += size
    
    def get_copy(self):
        return self.data["copy"]

    def add_multicast(self, size):
        self.data["multicast"] += size
    
    def get_multicast(self):
        return self.data["multicast"]

    def add_cube(self, core_id, count):
        if core_id not in self.data["cube_flops"]:
            self.data["cube_flops"][core_id] = 0
        self.data["cube_flops"][core_id] += count

    def get_cube_occupancy(self):
        if len(self.data["cube_flops"]) == 0:
            return 0
        
        max_flops = max([self.data["cube_flops"][core_id] for core_id in self.data["cube_flops"]])
        avg_flops = sum([self.data["cube_flops"][core_id] for core_id in self.data["cube_flops"]]) / len(self.data["cube_flops"])

        return avg_flops / (max_flops + np.finfo(float).eps)
    
    def get_total_cube(self):
        return sum(self.data["cube_flops"][core_id] for core_id in self.data["cube_flops"] )

    def add_vector(self, core_id, count):
        if core_id not in self.data["vector_flops"]:
            self.data["vector_flops"][core_id] = 0
        self.data["vector_flops"][core_id] += count

    def get_vector(self):
        return sum(self.data["vector_flops"][core_id] for core_id in self.data["vector_flops"])

    def merge(self, other: "Stats"):
        self.data["reads"] += other.data["reads"]
        self.data["writes"] += other.data["writes"]
        self.data["copy"] += other.data["copy"]
        self.data["multicast"] += other.data["multicast"]

        for core_id in other.data["cube_flops"]:
            self.add_cube(core_id, other.data["cube_flops"][core_id])

        for core_id in other.data["vector_flops"]:
            self.add_vector(core_id, other.data["vector_flops"][core_id])

    def log_stats(self, uid, layer_type, node_id, expected=None, dims=None, tile_size=None):
        if expected:
            expected_str = []
            for key in expected:
                if key == "flops":
                    val = flops_to_str(expected[key])
                else:
                    val = byte_to_str(expected[key]) 
                expected_str.append(f"{key}: {val}")


        msg = ""
        msg += f"{uid} {layer_type}"
        msg += f"\n\tdims: {list(dims) if dims else 'N/A'}"
        msg += f"\ttile_size: {list(tile_size) if tile_size else 'N/A'}"
        msg += f"\n\tEXPECTED: "
        msg += f"{' '.join(expected_str)}" if expected else 'N/A'
        msg += f"\n\tACTUAL:"
        msg += f"\treads: {byte_to_str(self.get_reads())}"
        msg += f"\twrites: {byte_to_str(self.get_writes())}"
        msg += f"\tcopy: {byte_to_str(self.get_copy())}"
        msg += f"\tmulticast: {byte_to_str(self.get_multicast())}"
        msg += f"\tcube: {flops_to_str(self.get_total_cube())}"
        msg += f"\tvector: {flops_to_str(self.get_vector())}"
        msg += f"\n\tmemory_overhead: {((self.get_reads() + self.get_writes()) / (expected['reads'] + expected['writes'] + np.finfo(float).eps) if expected else 0):<.2f}"
        msg += f"\tcube_occupancy: {self.get_cube_occupancy():.2f}"
        msg += "\n"
        
        with open(f"logs/core_level/node{node_id}.log", "a") as f:
            f.write(msg + "\n")
