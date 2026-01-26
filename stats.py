
import csv
import logging 
import os 

from utils import byte_to_str, mac_to_str

class NodeStats:
    def __init__(self, iter=None):
        if iter:
            self.iter = iter
            self.stats[iter] = {}
        else:
            self.iter = None
            self.stats = {}

    def new_iter(self, iter_id):
        self.iter = iter_id
        self.stats[self.iter] = {}
        
    def append(self, uid, operation, memory_footprint, num_ops, hbm_reads, network_data, comm_group, dims):
        self.stats[self.iter][uid] = {
            "operation": operation,
            "memory_footprint": memory_footprint, 
            "num_ops": num_ops, 
            "hbm_reads": hbm_reads, 
            "network_data": network_data, 
            "comm_group": "N/A" if comm_group is None else comm_group,
            "dims": dims
        }

    def get_stats(self, uid):
        return self.stats[self.iter][uid]

    def merge(self, other_stats):
        assert self.iter == other_stats.iter, "Cannot merge stats from different iterations"
        for uid in other_stats.stats[other_stats.iter]:
            self.stats[self.iter][uid] = other_stats.stats[other_stats.iter][uid]

    def sumUp(self):
        memory_footprint = sum([self.stats[self.iter][uid]["memory_footprint"] for uid in self.stats[self.iter]])
        num_ops = sum([self.stats[self.iter][uid]["num_ops"] for uid in self.stats[self.iter]])
        hbm_reads = sum([self.stats[self.iter][uid]["hbm_reads"] for uid in self.stats[self.iter]])
        network_data = sum([self.stats[self.iter][uid]["network_data"] for uid in self.stats[self.iter]])
        return memory_footprint, num_ops, hbm_reads, network_data

    def write_to_csv(self, fname):
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
            
        with open(fname, "w") as f:
            fieldnames = ["uid", "operation", "memory_footprint", "num_ops", "hbm_reads", "network_data", "comm_group", "dims"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writerows([{"uid":"uid", "operation":"operation", "memory_footprint":"memory_footprint (B)", "num_ops":"num_ops (MAC)", "hbm_reads":"hbm_reads (B)", "network_data":"network_data (B)", "comm_group":"comm. group", "dims": "Dimensions"}])
            writer.writerows([{"uid": uid} | self.stats[self.iter][uid] for uid in self.stats[self.iter]])

    def summarize(self):
        memory_footprint, num_ops, hbm_reads, network_data = self.sumUp()

        logging.info("--------- Summary per device -----------")
        logging.info("memory_footprint: {}".format(byte_to_str(memory_footprint)))
        logging.info("num_ops: {}".format(mac_to_str(num_ops)))
        logging.info("hbm_reads: {}".format(byte_to_str(hbm_reads)))
        logging.info("network_data: {}".format(byte_to_str(network_data)))
        logging.info("--------- End of Summary -----------")

        return memory_footprint, num_ops, hbm_reads, network_data
