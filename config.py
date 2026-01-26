
import json
from copy import deepcopy
import logging 
from collections import OrderedDict
import numpy as np

def get_itemids_from_bucketid(bucket_id, n_items, n_buckets):
    assert bucket_id < n_buckets

    n_items_per_bucket_low = n_items // n_buckets
    n_items_per_bucket_high = n_items_per_bucket_low + 1

    n_buckets_with_high = n_items - n_items_per_bucket_low * n_buckets
    n_buckets_with_low = n_buckets - n_buckets_with_high

    if bucket_id < n_buckets_with_high:
        item_ids = list(range(bucket_id*n_items_per_bucket_high, (bucket_id+1)*n_items_per_bucket_high))
    else:
        start_expert_id = n_buckets_with_high * n_items_per_bucket_high + (bucket_id - n_buckets_with_high) * n_items_per_bucket_low
        item_ids = list(range(start_expert_id, start_expert_id + n_items_per_bucket_low))

    return item_ids


def get_bucketid_from_itemid(item_id, n_items, n_buckets):
    assert np.all(item_id < n_items)

    n_items_per_bucket_low = n_items // n_buckets
    n_items_per_bucket_high = n_items_per_bucket_low + 1

    n_buckets_with_high = n_items - n_items_per_bucket_low * n_buckets
    n_buckets_with_low = n_buckets - n_buckets_with_high

    boundary_id = n_items_per_bucket_high * n_buckets_with_high

    if isinstance(item_id, np.ndarray):
        bucket_id = np.where(item_id < boundary_id, item_id // n_items_per_bucket_high, n_buckets_with_high + (item_id - boundary_id) // n_items_per_bucket_low)
    else:
        if item_id < boundary_id:
            bucket_id = item_id // n_items_per_bucket_high
        else:
            bucket_id = n_buckets_with_high + (item_id - boundary_id) // n_items_per_bucket_low

    return bucket_id

class DistInfo:
    def __init__(self, global_cfg, rank, num_nodes, dp_attn, dp_ffn, tp_attn, tp_ffn, pp, sp, ep, expert_workload_model, ranks, attn_comm_groups, ffn_comm_groups, n_redundant_shared_exp, moe_comm) -> None:
        self.global_cfg = global_cfg
        self.rank = rank 
        self.num_nodes = num_nodes
        self.dp_attn = dp_attn
        self.dp_ffn = dp_ffn
        self.tp_attn = tp_attn
        self.tp_ffn = tp_ffn
        self.pp = pp
        self.sp = sp
        self.ep = ep
        self.expert_workload_model = expert_workload_model
        self.moe_comm = moe_comm
        
        self.rank_dp_attn = ranks["dp_attn"]
        self.rank_dp_ffn = ranks["dp_ffn"]
        self.rank_tp_attn = ranks["tp_attn"]
        self.rank_tp_ffn = ranks["tp_ffn"]
        self.rank_pp = ranks["pp"]
        self.rank_sp = ranks["sp"]
        self.rank_ep = ranks["ep"]

        self.attn_comm_groups = attn_comm_groups
        self.ffn_comm_groups = ffn_comm_groups

        # list of node ids in the same dp_attn cluster as self.rank
        self.dp_attn_cluster = [k for k,v in global_cfg.ranks["dp_attn"].items() if v == self.rank_dp_attn]

        # map batch id to dp rank for attention and ffn layers
        # structure: {"attn": {batch_id: dp_rank}, "ffn": {batch_id: dp_rank}}
        self.batch_map = None

        # stores which batch is mapped to which shared expert copy
        # structure: {batch_id: node_id of shared expert}
        self.batch_to_shared_exp = None 

        # n_redundant_shared_exp is the number of redundant shared expert copies in the system.
        self.n_redundant_shared_exp = n_redundant_shared_exp
        assert num_nodes % n_redundant_shared_exp == 0, "Number of nodes must be divisible by n_redundant_shared_exp"

        if self.ep == self.num_nodes:
            # each redundant shared expert copy will process samples from a number of nodes that is equal to num_nodes // n_redundant_shared_exp
            shared_exp_cluster_size = num_nodes // n_redundant_shared_exp

            # shared_expert_ranks is a list of ranks that are assigned to each redundant shared expert copy. only these ranks will keep a copy of the redundant shared expert.
            self.shared_expert_ranks = [i*shared_exp_cluster_size for i in range(n_redundant_shared_exp)]
        elif self.tp_ffn == self.num_nodes:
            self.shared_expert_ranks = ffn_comm_groups["tp_ffn"]
        else:
            raise NotImplementedError("Only full EP or full TP is supported for FFN blocks currently.")
        
    # every DP cluster has a master node that is responsible for broadcasting. 
    # this function returns True if the current rank is the master of its DP cluster.
    def is_dp_master(self):
        return self.rank == self.dp_attn_cluster[0]

    def get_expert_mapping(self, n_experts):
        return {expert_id: get_bucketid_from_itemid(expert_id, n_experts, self.ep) for expert_id in range(n_experts)}

    def batch_mapping(self, bsz):
        assert bsz >= self.dp_attn and bsz >= self.dp_ffn, "Batch size must be larger than dp_attn and dp_ffn"

        self.batch_map = {"attn": {}, "ffn": {}}

        for rank_dp_attn in range(self.dp_attn):
            self.batch_map["attn"].update({batch_id:rank_dp_attn for batch_id in get_itemids_from_bucketid(rank_dp_attn, bsz, self.dp_attn)})

        for rank_dp_ffn in range(self.dp_ffn):
            self.batch_map["ffn"].update({batch_id:rank_dp_ffn for batch_id in get_itemids_from_bucketid(rank_dp_ffn, bsz, self.dp_ffn)})

        # self.batch_to_shared_exp = {batch_id: self.nearest_shared_expert(batch_id) for batch_id in range(bsz)}
        self.batch_to_shared_exp = {}
        for batch_id in range(bsz):
            bucketid = get_bucketid_from_itemid(batch_id, bsz, self.n_redundant_shared_exp)
            self.batch_to_shared_exp[batch_id] = self.shared_expert_ranks[bucketid]

    def get_dp_rank_from_batchids(self, batch_ids, layer_type):
        assert layer_type in ["attn", "ffn"], "layer_type must be either 'attn' or 'ffn'"
        return [self.batch_map[layer_type][batch_id] for batch_id in batch_ids]

    def get_local_batchids(self, layer_type):
        ''' batch ids that are mapped to the current node's dp cluster '''
        assert layer_type in ["attn", "ffn"], "layer_type must be either 'attn' or 'ffn'"
        rank_dp = self.rank_dp_attn if layer_type == "attn" else self.rank_dp_ffn
        return [batch_id for batch_id, r in self.batch_map[layer_type].items() if r == rank_dp]

    def get_dp_master(self, dp_rank, layer_type):
        assert layer_type in ["attn", "ffn"], "layer_type must be either 'attn' or 'ffn'"
        dp_cluster = [k for k,v in self.global_cfg.ranks["dp_"+layer_type].items() if v == dp_rank]
        return dp_cluster[0]

    # def nearest_shared_expert(self, batch_id):
    #     batch_dp_rank = self.batch_map["attn"][batch_id]
    #     master_dp_rank = self.get_dp_master(batch_dp_rank, "attn")

    #     nearest_node = self.shared_expert_ranks[0]
    #     dp_cluster_size = len(self.dp_attn_cluster)

    #     for shared_exp_rank in self.shared_expert_ranks[1:]:
    #         if shared_exp_rank < master_dp_rank + dp_cluster_size:
    #             nearest_node = shared_exp_rank

    #     return nearest_node

class SystemConfig:
    def __init__(self) -> None:
        return

    def construct(self):
        for val in [self.dp_attn, self.dp_ffn, self.tp_attn, self.tp_ffn, self.pp, self.sp, self.ep]:
            assert self.num_nodes % val == 0
        assert self.num_nodes == self.dp_attn * self.tp_attn * self.sp * self.pp, "Number of nodes is not equal to the parallelization factors for Attention blocks"
        assert self.num_nodes == self.dp_ffn * self.tp_ffn * self.ep * self.pp, "Number of nodes is not equal to the parallelization factors for FFN blocks"

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
        self.ranks = attn_ranks
        self.ranks.update(ffn_ranks)

        for rank in range(self.num_nodes):
            logging.info("rank:{}\t rank_pp:{}\t rank_dp_attn:{}\t rank_sp:{}\t rank_tp_attn:{}\t pp_comm_group:{}\t dp_attn_comm_group:{}\t sp_comm_group:{}\t tp_attn_comm_group:{}"
                         .format(rank, self.ranks["pp"][rank], self.ranks["dp_attn"][rank], self.ranks["sp"][rank], self.ranks["tp_attn"][rank], 
                                 attn_comm_groups["pp"][rank], attn_comm_groups["dp_attn"][rank], attn_comm_groups["sp"][rank], attn_comm_groups["tp_attn"][rank]))

        for rank in range(self.num_nodes):
            logging.info("rank:{}\t rank_pp:{}\t rank_dp_ffn:{}\t rank_ep:{}\t rank_tp_ffn:{}\t pp_comm_group:{}\t dp_ffn_comm_group:{}\t ep_comm_group:{}\t tp_ffn_comm_group:{}"
                         .format(rank, self.ranks["pp"][rank], self.ranks["dp_ffn"][rank], self.ranks["ep"][rank], self.ranks["tp_ffn"][rank],
                                 ffn_comm_groups["pp"][rank], ffn_comm_groups["dp_ffn"][rank], ffn_comm_groups["ep"][rank], ffn_comm_groups["tp_ffn"][rank]))

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
