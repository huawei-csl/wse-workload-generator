import logging 

from utils import byte_to_str, flops_to_str

class Stats:
    def __init__(self, core_ids=None):
        self.data = {
            "reads": 0,
            "writes": 0,
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

    def add_cube(self, core_id, count):
        if core_id not in self.data["cube_flops"]:
            self.data["cube_flops"][core_id] = 0
        self.data["cube_flops"][core_id] += count

    def get_cube_occupancy(self):
        max_flops = max([self.data["cube_flops"][core_id] for core_id in self.data["cube_flops"]])
        avg_flops = sum([self.data["cube_flops"][core_id] for core_id in self.data["cube_flops"]]) / len(self.data["cube_flops"])

        return avg_flops / max_flops
    
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

        for core_id in other.data["cube_flops"]:
            self.add_cube(core_id, other.data["cube_flops"][core_id])

        for core_id in other.data["vector_flops"]:
            self.add_vector(core_id, other.data["vector_flops"][core_id])

    def print_stats(self, uid, expected=None, dims=None, tile_size=None):
        if expected:
            # expected_str = (
            #     f"input: {byte_to_str(expected['input0_size']):<15}"
            #     f"weight: {byte_to_str(expected['input1_size']):<15}"
            #     f"output: {byte_to_str(expected['output_size']):<15}"
            #     f"read: {byte_to_str(expected["reads"]):<15}"
            #     f"write: {byte_to_str(expected["writes"]):<15}"
            #     f"flops: {flops_to_str(expected['flops']):<15}"
            # )

            expected_str = []
            for key in expected:
                if key == "flops":
                    val = flops_to_str(expected[key])
                else:
                    val = byte_to_str(expected[key]) 
                expected_str.append(f"{key}: {val:<15}")
        else:
            expected_str = "N/A\t"

        print(
            f"Stats for {uid:<40}\t"
            f"dims: {list(dims) if dims else "N/A"}\t"
            f"tile_size: {list(tile_size) if tile_size else "N/A"}\t"
            f"-- EXPECTED --\t"
            f"{'\t'.join(expected_str)}"
            f"-- ACTUAL --\t"
            f"reads: {byte_to_str(self.get_reads()):<15}"
            f"writes: {byte_to_str(self.get_writes()):<15}"
            f"cube: {flops_to_str(self.get_total_cube()):<15}"
            f"vector: {flops_to_str(self.get_vector()):<15}"
            f"memory_overhead: {((self.get_reads() + self.get_writes()) / (expected["reads"]+expected["writes"]) if expected else 0):<10.2f}"
            f"cube_occupancy: {self.get_cube_occupancy():<10.2f}"
        )