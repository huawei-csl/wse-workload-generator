
import json
import logging 
from collections import OrderedDict
from src.node_level.common.dist_info import DistInfo

class SystemConfig:
    def __init__(self) -> None:
        return

    def construct(self):
        for val in [self.dp_attn, self.dp_ffn, self.tp_attn, self.tp_ffn, self.pp, self.sp, self.ep]:
            assert self.num_nodes % val == 0
        assert self.num_nodes == self.dp_attn * self.tp_attn * self.sp * self.pp, f"Number of nodes {self.num_nodes} is not equal to the parallelization factors for Attention blocks: {self.dp_attn} x {self.tp_attn} x {self.sp} x {self.pp}"
        assert self.num_nodes == self.dp_ffn * self.tp_ffn * self.ep * self.pp, f"Number of nodes {self.num_nodes} is not equal to the parallelization factors for FFN blocks: {self.dp_ffn} x {self.tp_ffn} x {self.ep} x {self.pp}"

        if self.ep > 1:
            assert self.dp_ffn == 1 and self.tp_ffn == 1, "If EP is used, do not use DP or TP for FFN (can still use them for ATTN)"

        assert self.dp_ffn == 1, "Currently, DP for FFN is not supported"
        assert self.ep == self.num_nodes or self.tp_ffn == self.num_nodes, "Currently, either full EP or full TP is supported"
        if self.tp_ffn == self.num_nodes: 
            assert self.n_redundant_shared_exp == 1, "If full TP is used for FFN, n_redundant_shared_exp must be 1"

        attn_par_degrees = {"tp_attn": self.tp_attn, "sp": self.sp, "dp_attn": self.dp_attn, "pp":self.pp}
        attn_comm_groups, attn_ranks = get_comm_groups(self.num_nodes, attn_par_degrees)
        self.attn_comm_groups = attn_comm_groups
        
        ffn_par_degrees = {"tp_ffn": self.tp_ffn, "ep": self.ep, "dp_ffn": self.dp_ffn, "pp":self.pp}
        ffn_comm_groups, ffn_ranks = get_comm_groups(self.num_nodes, ffn_par_degrees)
        self.ffn_comm_groups = ffn_comm_groups

        dense_par_degrees = {"tp_dense": self.tp_attn * self.sp, "dp_dense": self.dp_attn, "pp":self.pp}
        dense_comm_groups, dense_ranks = get_comm_groups(self.num_nodes, dense_par_degrees)
        self.dense_comm_groups = dense_comm_groups

        self.ranks = attn_ranks | ffn_ranks | dense_ranks

        for rank in range(self.num_nodes):
            logging.info("rank:{}\t rank_pp:{}\t rank_dp_attn:{}\t rank_sp:{}\t rank_tp_attn:{}\t pp_comm_group:{}\t dp_attn_comm_group:{}\t sp_comm_group:{}\t tp_attn_comm_group:{}"
                         .format(rank, self.ranks["pp"][rank], self.ranks["dp_attn"][rank], self.ranks["sp"][rank], self.ranks["tp_attn"][rank], 
                                 attn_comm_groups["pp"][rank], attn_comm_groups["dp_attn"][rank], attn_comm_groups["sp"][rank], attn_comm_groups["tp_attn"][rank]))

        for rank in range(self.num_nodes):
            logging.info("rank:{}\t rank_pp:{}\t rank_dp_ffn:{}\t rank_ep:{}\t rank_tp_ffn:{}\t pp_comm_group:{}\t dp_ffn_comm_group:{}\t ep_comm_group:{}\t tp_ffn_comm_group:{}"
                         .format(rank, self.ranks["pp"][rank], self.ranks["dp_ffn"][rank], self.ranks["ep"][rank], self.ranks["tp_ffn"][rank],
                                 ffn_comm_groups["pp"][rank], ffn_comm_groups["dp_ffn"][rank], ffn_comm_groups["ep"][rank], ffn_comm_groups["tp_ffn"][rank]))

        for rank in range(self.num_nodes):
            logging.info("rank:{}\t rank_pp:{}\t rank_dp_dense:{}\t rank_tp_dense:{}\t pp_comm_group:{}\t dp_dense_comm_group:{}\t tp_dense_comm_group:{}"
                         .format(rank, self.ranks["pp"][rank], self.ranks["dp_dense"][rank], self.ranks["tp_dense"][rank],
                                 dense_comm_groups["pp"][rank], dense_comm_groups["dp_dense"][rank], dense_comm_groups["tp_dense"][rank]))

    def from_args(self, 
                  num_nodes: int = 1, 
                  dp_attn: int = 1, 
                  dp_ffn: int = 1, 
                  tp_attn: int = 1, 
                  tp_ffn: int = 1, 
                  pp: int = 1, 
                  sp: int = 1, 
                  ep: int = 1, 
                  n_redundant_shared_exp: int = 1, 
                  expert_workload_model: str = "empirical_mmlu", 
                  moe_comm: str = "multicast"):
        
        assert expert_workload_model in ["identical", "uniform", "empirical_mmlu"], "expert_workload_model must be one of ['identical', 'uniform', 'empirical_mmlu']"
        assert moe_comm in ["alltoall", "multicast"], "moe_comm must be one of ['alltoall', 'multicast']"

        self.num_nodes = num_nodes
        self.dp_attn = dp_attn
        self.dp_ffn = dp_ffn
        self.tp_attn = tp_attn
        self.tp_ffn = tp_ffn
        self.pp = pp
        self.sp = sp
        self.ep = ep
        self.n_redundant_shared_exp = n_redundant_shared_exp
        self.expert_workload_model = expert_workload_model
        self.moe_comm = moe_comm

        self.construct()
        return self
    
    def from_json(self, fname, mode):
        with open(fname, "r") as f:
            cfg = json.load(f)[mode]
    
        self.num_nodes = cfg["num_nodes"]
        self.node_grid = cfg["node_grid"]
        self.core_grid = cfg["core_grid"]
        assert self.num_nodes == self.node_grid[0] * self.node_grid[1], "num_nodes does not match node_grid"
        
        self.dp_attn = cfg["dp_attn"]
        self.dp_ffn = cfg["dp_ffn"]
        self.tp_attn = cfg["tp_attn"]
        self.tp_ffn = cfg["tp_ffn"]
        self.pp = cfg["pp"]
        self.sp = cfg["sp"]
        self.ep = cfg["ep"]
        self.n_redundant_shared_exp = cfg["n_redundant_shared_exp"]
        self.expert_workload_model = cfg["expert_workload_model"]
        self.moe_comm = cfg["moe_comm"]

        self.construct()
        return self

    def get_dist_info(self, rank):
        return DistInfo(
            self,
            rank=rank,
            num_nodes = self.num_nodes,
            dp_attn = self.dp_attn,
            dp_ffn = self.dp_ffn,
            tp_attn = self.tp_attn,
            tp_ffn = self.tp_ffn,
            pp = self.pp,
            sp = self.sp,
            ep = self.ep,
            expert_workload_model = self.expert_workload_model,
            ranks =  {k: self.ranks[k][rank]  for k in self.ranks},
            attn_comm_groups = {k: self.attn_comm_groups[k][rank]  for k in self.attn_comm_groups}, 
            ffn_comm_groups = {k: self.ffn_comm_groups[k][rank]  for k in self.ffn_comm_groups},
            dense_comm_groups = {k: self.dense_comm_groups[k][rank]  for k in self.dense_comm_groups},
            n_redundant_shared_exp = self.n_redundant_shared_exp,
            moe_comm=self.moe_comm
        )

'''
This function calculates the communication groups for a given set of parallelization strategies. 
A comm. group defines a group of devices that must perform a CC operation (e.g., allreduce).

Suppose you have 8 devices (ranks: [0, 1, 2, 3, 4, 5, 6, 7])
Suppose you want each 4 neighboring devices perform tensor parallelism (0,1,2,3-> tp_cluster_0, 4,5,6,7-> tp_cluster_1)
On top of this, suppose you want to perform sequence parallelism (first half of KV-cache -> tp_cluster0, other half of KV-cache -> tp_cluster_1)

This function will return:
rank:0   rank_tp:0       rank_sp:0       tp_comm_group:[0, 1, 2, 3]      sp_comm_group:[0, 4]
rank:1   rank_tp:1       rank_sp:0       tp_comm_group:[0, 1, 2, 3]      sp_comm_group:[1, 5]
rank:2   rank_tp:2       rank_sp:0       tp_comm_group:[0, 1, 2, 3]      sp_comm_group:[2, 6]
rank:3   rank_tp:3       rank_sp:0       tp_comm_group:[0, 1, 2, 3]      sp_comm_group:[3, 7]
rank:4   rank_tp:0       rank_sp:1       tp_comm_group:[4, 5, 6, 7]      sp_comm_group:[0, 4]
rank:5   rank_tp:1       rank_sp:1       tp_comm_group:[4, 5, 6, 7]      sp_comm_group:[1, 5]
rank:6   rank_tp:2       rank_sp:1       tp_comm_group:[4, 5, 6, 7]      sp_comm_group:[2, 6]
rank:7   rank_tp:3       rank_sp:1       tp_comm_group:[4, 5, 6, 7]      sp_comm_group:[3, 7]

This means that for TP we must perform Allreduce among [0, 1, 2, 3] and [4, 5, 6, 7]
For SP, we must perform Allreduce among [0, 4], [1, 5], [2, 6], and [3, 7]

Arguments:
    num_nodes: number of devices
    par_degrees: An ordered dict whose keys are types of parallelism and values are cluster size for each type of parallelism. The order of the dict defines the order of hierarchy.

'''
def get_comm_groups(num_nodes: int, par_degrees: OrderedDict):
    comm_groups = OrderedDict({key: {} for key in par_degrees})
    ranks = OrderedDict({key: {} for key in par_degrees})
    for rank in range(num_nodes):
        subcluster_size = 1
        cluster_size = 1
        rank_offset = 0

        for par_type in par_degrees:
            par_degree = par_degrees[par_type]
            ranks[par_type][rank] = (rank // cluster_size) % par_degree
            cluster_size = subcluster_size*par_degree
            cluster_offset = (rank // cluster_size) * cluster_size
            comm_groups[par_type][rank] = sorted([(rank_offset + j*subcluster_size) % cluster_size + cluster_offset for j in range(par_degree)])
            rank_offset = rank_offset + ranks[par_type][rank]*subcluster_size
            subcluster_size = cluster_size

    return comm_groups, ranks


if __name__=="__main__":
    num_nodes = 16
    par_degrees = OrderedDict({"tp": 4, "sp": 2, "dp": 2})

    res = 1
    for par_type in par_degrees:
        res = res * par_degrees[par_type]
    assert res == num_nodes, "number of nodes does not match the parallelization degrees"

    comm_groups, ranks = get_comm_groups(num_nodes, par_degrees)

    for rank in range(num_nodes):
        print("rank:{}\t rank_tp:{}\t rank_sp:{}\t rank_dp:{}\t tp_comm_group:{}\t sp_comm_group:{}\t dp_comm_group:{}"
              .format(rank, ranks["tp"][rank], ranks["sp"][rank], ranks["dp"][rank], comm_groups["tp"][rank], comm_groups["sp"][rank], comm_groups["dp"][rank]))
