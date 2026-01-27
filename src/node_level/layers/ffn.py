import logging

from src.node_level.common.tensor import Tensor
from src.node_level.common.compute_graph import reset_compute_graph
from src.node_level.layers.linear import Linear
from src.node_level.layers.allreduce import Allreduce
from src.node_level.layers.add import Add

from src.node_level.common.config import SystemConfig
from src.node_level.common.stats import NodeStats

from src.node_level.common.utils import intceil, dtype_to_byte

# Each expert is an instance of this layer
class FFN:
    def __init__(self, uid, hidden_size, intermediate_size, dist_info, dtype, is_dense_layer=False) -> None:
        super().__init__()
        logging.info("Creating FFN layer {}".format(uid))
        self.uid = uid
        self.rank = dist_info.rank
        self.dist_info = dist_info
        self.dtype = dtype
        
        if is_dense_layer:
            self.par_factor = dist_info.tp_attn * dist_info.sp
            self.comm_group = dist_info.dense_comm_groups["tp_dense"]
        else:
            self.par_factor = dist_info.tp_ffn
            self.comm_group = dist_info.ffn_comm_groups["tp_ffn"]

        inter_size_per_node = intceil(intermediate_size/self.par_factor) 

        self.ops = {}
        self.ops["up"] = Linear(uid+"_up", self.rank, hidden_size, inter_size_per_node, dtype)
        self.ops["gate"] = Linear(uid+"_gate", self.rank, hidden_size, inter_size_per_node, dtype)
        self.ops["down"] = Linear(uid+"_down", self.rank, inter_size_per_node, hidden_size, dtype)
        
        # in case we use TP for experts
        if self.par_factor > 1:
            self.ops["allreduce"] = Allreduce(uid+"_ar", self.rank, hidden_size, self.comm_group, dtype)

        self._stats = NodeStats()

    def forward(self, x, stats):
        self._stats.new_iter(stats.iter)

        x1 = self.ops["up"].forward(x, stats=self._stats)
        x2 = self.ops["gate"].forward(x, stats=self._stats)
        
        x_add = Add(self.uid+"_add", self.rank, x1.dims, dtype=self.dtype).forward(x1, x2, stats=self._stats)

        y = self.ops["down"].forward(x_add, stats=self._stats)

        if self.par_factor > 1:
            y = self.ops["allreduce"].forward(y, stats=self._stats)
        
        stats.merge(self._stats)
        return y

    def memory_footprint(self, bsz=None, ctx_len=None):
        mem_size = sum([self.ops[opname].memory_footprint() for opname in self.ops])
        return mem_size # in bytes



if __name__ == "__main__":
    reset_compute_graph()

    bsz = 32

    dp_attn = 2
    dp_ffn = 1
    tp_attn = 2
    tp_ffn = 1
    ep = 8
    sp = 2
    
    hidden_size = 7168
    intermediate_size = 18432
    dtype = "fp16"
    is_dense_layer = True 

    seqlen = 1

    num_nodes = dp_attn * tp_attn * sp
    decode_cfg = SystemConfig().from_args(
        num_nodes=num_nodes, 
        dp_attn=dp_attn,
        dp_ffn=dp_ffn,
        tp_attn=tp_attn,
        tp_ffn=tp_ffn,
        sp=sp,
        ep=ep
    )

    for rank in range(num_nodes):
        dist_info = decode_cfg.get_dist_info(rank)
        dist_info.batch_mapping(bsz)

        stats = NodeStats()
        stats.new_iter(iter_id=0)

        batch_ids = dist_info.get_local_batchids("attn")
        local_bsz = len(batch_ids)

        op = FFN(f"{rank}ffn_0", hidden_size, intermediate_size, dist_info, dtype, is_dense_layer=is_dense_layer)

        x = Tensor("input", rank, [local_bsz, seqlen, hidden_size])

        y = op.forward(x, stats=stats)

        assert y.dims == [local_bsz, seqlen, hidden_size]

        par_factor = tp_attn * sp 
        local_inter_size = intceil(intermediate_size / par_factor)
        ffn_comm_size = par_factor

        expected = {
            "memory_footprint": 3 * hidden_size * local_inter_size * dtype_to_byte(dtype), # weights only
            "num_ops": 3 * local_bsz * seqlen * hidden_size * local_inter_size, # in terms of MACs
            "hbm_reads": 3 * hidden_size * local_inter_size * dtype_to_byte(dtype), # weights only
            "network_data": 4 * intceil( (local_bsz * seqlen * hidden_size) / ffn_comm_size) * (ffn_comm_size - 1) * dtype_to_byte(dtype) # allreduce for tp_ffn
        }

        op_mem_foot, op_num_ops, op_hbm_reads, op_net_data = op._stats.sumUp()

        assert expected["memory_footprint"] == op_mem_foot, f"Expected memory_footprint {expected['memory_footprint']}, got {op_mem_foot}"
        assert expected["num_ops"] == op_num_ops, f"Expected num_ops {expected['num_ops']}, got {op_num_ops}"
        assert expected["hbm_reads"] == op_hbm_reads, f"Expected hbm_reads {expected['hbm_reads']}, got {op_hbm_reads}"
        assert expected["network_data"] == op_net_data, f"Expected network_data {expected['network_data']}, got {op_net_data}"