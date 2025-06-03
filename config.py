
class SystemConfig:
    def __init__(self, rank, num_nodes, dp_attn=1, dp_ffn=1, tp_attn=1, tp_ffn=1, pp=1, sp=1, ep=1) -> None:
        assert rank < num_nodes
        assert num_nodes % dp_attn == 0 and num_nodes % dp_ffn == 0 and num_nodes % tp_attn == 0 and num_nodes % tp_ffn == 0 and num_nodes % pp == 0 and num_nodes % sp == 0 and num_nodes % ep == 0 
        assert num_nodes == dp_attn * tp_attn * sp * pp, "Number of nodes is not equal to the parallelization factors for Attention blocks"
        assert num_nodes == dp_ffn * tp_ffn * ep * pp, "Number of nodes is not equal to the parallelization factors for FFN blocks"

        self.rank = rank
        self.num_nodes = num_nodes
        self.dp_attn = dp_attn
        self.dp_ffn = dp_ffn
        self.tp_attn = tp_attn
        self.tp_ffn = tp_ffn
        self.pp = pp
        self.sp = sp
        self.ep = ep

        self.get_ranks()
        
    def get_ranks(self):
        cluster_size = self.num_nodes
        rank = self.rank

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

        # print(rank_dp_attn, rank_pp, rank_tp_attn, rank_sp)

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

        # print(rank_dp_ffn, rank_pp, rank_tp_ffn, rank_ep)
