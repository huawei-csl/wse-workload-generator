
import json
from copy import deepcopy


class SystemConfig:
    def __init__(self, fname, mode) -> None:
        self.load_from_json(fname, mode)

        for val in [self.dp_attn, self.dp_ffn, self.tp_attn, self.tp_ffn, self.pp, self.sp, self.ep]:
            assert self.num_nodes % val == 0
        assert self.num_nodes == self.dp_attn * self.tp_attn * self.sp * self.pp, "Number of nodes is not equal to the parallelization factors for Attention blocks"
        assert self.num_nodes == self.dp_ffn * self.tp_ffn * self.ep * self.pp, "Number of nodes is not equal to the parallelization factors for FFN blocks"

        if self.ep > 1:
            assert self.dp_ffn == 1, "If EP is used, do not use DP for FFN (can still use it for ATTN)"
        
    def get_ranks(self, rank):                
        assert rank < self.num_nodes

        # ATTN ranks
        cluster_size = self.num_nodes
        self.rank = rank 

        cluster_size = cluster_size // self.pp 
        self.rank_pp = rank // cluster_size
        rank = rank % cluster_size

        cluster_size = cluster_size // self.dp_attn
        self.rank_dp_attn = rank // cluster_size
        rank = rank % cluster_size

        cluster_size = cluster_size // self.tp_attn
        self.rank_tp_attn = rank // cluster_size
        rank = rank % cluster_size

        cluster_size = cluster_size // self.sp
        self.rank_sp = rank // cluster_size
        rank = rank % cluster_size

        # FFN ranks
        cluster_size = self.num_nodes
        rank = self.rank

        cluster_size = cluster_size // self.pp 
        self.rank_pp = rank // cluster_size
        rank = rank % cluster_size

        cluster_size = cluster_size // self.dp_ffn
        self.rank_dp_ffn = rank // cluster_size
        rank = rank % cluster_size

        cluster_size = cluster_size // self.tp_ffn
        self.rank_tp_ffn = rank // cluster_size
        rank = rank % cluster_size

        cluster_size = cluster_size // self.ep
        self.rank_ep = rank // cluster_size
        rank = rank % cluster_size

    def load_from_json(self, fname, mode):
        with open(fname, "r") as f:
            cfg = json.load(f)[mode]
    
        self.num_nodes = cfg["num_nodes"]
        self.dp_attn = cfg["dp_attn"]
        self.dp_ffn = cfg["dp_ffn"]
        self.tp_attn = cfg["tp_attn"]
        self.tp_ffn = cfg["tp_ffn"]
        self.pp = cfg["pp"]
        self.sp = cfg["sp"]
        self.ep = cfg["ep"]
        self.expert_workload_model = cfg["expert_workload_model"]
