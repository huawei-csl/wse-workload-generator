
import logging

from typing import List


class Barrier:
    def __init__(self, uid, nodes: List[int]) -> None:
        super().__init__()
        logging.debug("Barrier layer {} for nodes {}".format(uid, nodes))

        self.uid = uid
        self.nodes = nodes

    def forward(self, stats=None):
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims()

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        stats.append(self.uid, "Barrier", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.nodes, dims=dims)

    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def get_dims(self):
        return "N/A"
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self):
        return 0
