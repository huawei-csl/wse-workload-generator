
import numpy as np
import logging

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
    def __init__(self, global_cfg, rank, num_nodes, dp_attn, dp_ffn, tp_attn, tp_ffn, pp, sp, ep, expert_workload_model, ranks, attn_comm_groups, ffn_comm_groups, dense_comm_groups, n_redundant_shared_exp, moe_comm) -> None:
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
        self.dense_comm_groups = dense_comm_groups
        
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
        # assert num_nodes % n_redundant_shared_exp == 0, "Number of nodes must be divisible by n_redundant_shared_exp"

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
        
        for rank in self.shared_expert_ranks:
            batch_ids = [batch_id for batch_id, shared_expert_rank in self.batch_to_shared_exp.items() if shared_expert_rank == rank]
            logging.debug("{} samples are mapped to shared expert on rank {}: {}".format(len(batch_ids), rank, batch_ids))

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
