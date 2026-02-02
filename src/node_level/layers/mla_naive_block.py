
import logging

from src.node_level.layers.linear import Linear
from src.node_level.layers.mla_naive import MLANaiveAttention
from src.node_level.layers.allreduce import Allreduce
from src.node_level.layers.allgather import AllGather

from src.node_level.common.utils import intceil

class MLANaiveBlock:
    def __init__(self, uid, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype) -> None:
        super().__init__()
        logging.info("Creating MLA naive layer {}".format(uid))

        self.uid = uid
        self.dist_info = dist_info
        self.next_layer = next_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.rank = dist_info.rank

        n_local_heads = intceil(n_heads / dist_info.tp_attn)

        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.ops = {}
        self.ops["wq_a"] = Linear(uid+"_wqa", self.rank, hidden_size, q_lora_rank, dtype)
        self.ops["wq_b"] = Linear(uid+"_wqb", self.rank, q_lora_rank, n_local_heads*qk_head_dim, dtype)
        self.ops["wkv_a"] = Linear(uid+"_wkva", self.rank, hidden_size, kv_lora_rank+qk_rope_head_dim, dtype)
        self.ops["wkv_b"] = Linear(uid+"_wkvb", self.rank, kv_lora_rank, n_local_heads*(qk_nope_head_dim + v_head_dim), dtype)

        self.ops["naive_attn"] = MLANaiveAttention(uid+"_naiveattn", n_local_heads, qk_head_dim, v_head_dim, dist_info, dtype)
        if dist_info.sp > 1:
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", self.rank, n_local_heads * v_head_dim, dist_info.attn_comm_groups["sp"], dtype)

        self.ops["wo"] = Linear(uid+"_wo", self.rank, n_local_heads*v_head_dim, hidden_size, dtype)
        if dist_info.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", self.rank, hidden_size, dist_info.attn_comm_groups["tp_attn"], dtype)

        if dist_info.moe_comm == "allgather":
            self.ag_dispatch = AllGather(uid+"_ag_disp", hidden_size, dist_info.num_nodes, dist_info, dtype)

    def forward(self, x, ctx_len, stats):
        raise NotImplementedError("Not yet implemented, ask for support")
    
        bsz, seqlen, _ = x.dims

        batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        local_bsz = len(batch_ids)

        assert ctx_len == 0, "Naive block should be used only for prefill"
        for opname in self.ops:
            if isinstance(self.ops[opname], MLANaiveAttention):
                self.ops[opname].forward(local_bsz, seqlen, ctx_len, stats=stats)
            else:
                self.ops[opname].forward(local_bsz*seqlen, stats=stats)

        # if the next layer is Dense, we need to multicast the output to all nodes that do not have a copy of the queries processed by this DP cluster
        # for example, if num_nodes = 16, dp_attn = 4, rank of this node is 0, then we multicast the queries to nodes [4,5,6,7,8,9,10,11,12,13,14,15] 
        if isinstance(self.next_layer, Dense):
            if self.dist_info.is_dp_master(): # only the master node in a DP cluster sends the multicast
                dst_nodes = [i for i in range(self.dist_info.num_nodes) if i not in self.dist_info.dp_attn_cluster] # all nodes not in this DP cluster
                Multicast(self.uid+"_multicast", vector_size=self.hidden_size*local_bsz*seqlen, src=self.dist_info.rank, dst=dst_nodes, dtype=self.dtype).forward(stats=stats)
            Barrier(self.uid+"_barrier", nodes=list(range(self.dist_info.num_nodes))).forward(stats=stats) # ensure all nodes have received the multicast before proceeding

        # if the next layer is MoE, we need to multicast the output to the experts selected by the gate for the current batch
        elif isinstance(self.next_layer, MoE):
            if self.dist_info.moe_comm == "allgather":
                self.ag_dispatch.forward(local_bsz*seqlen, stats=stats)
            elif self.dist_info.moe_comm == "multicast":
                # batch ids processed by this DP cluster
                batch_ids = list(range(self.dist_info.rank_dp_attn*local_bsz*seqlen, (self.dist_info.rank_dp_attn+1)*local_bsz*seqlen))
                for batch_id in batch_ids:
                    # get expert ids for this query
                    mapping = get_moe_gate_model().get_mapping_by_batchids(self.next_layer.uid, batch_id)
                    logging.debug("batch_id: {}, mapping: {}".format(batch_id, mapping))

                    # calculate with nodes the experts are located
                    dst_nodes = sorted([get_bucketid_from_itemid(expert_id, self.next_layer.n_experts, self.dist_info.ep) for expert_id in mapping.tolist()])

                    # remove repeating nodes from dst_nodes
                    dst_nodes = list(dict.fromkeys(dst_nodes))

                    Multicast(self.uid+"_multicast_"+str(batch_id), vector_size=self.hidden_size, src=self.dist_info.rank, dst=dst_nodes, dtype=self.dtype).forward(stats=stats)
                Barrier(self.uid+"_barrier", nodes=list(range(self.dist_info.num_nodes))).forward(stats=stats) # ensure all nodes have received the multicast before proceeding
            else:
                raise NotImplementedError("MoE communication method {} not implemented".format(self.dist_info.moe_comm))
        
        else:
            raise NotImplementedError("Next layer type {} not implemented".format(type(self.next_layer)))
        
    def memory_footprint(self, bsz, ctx_len):
        batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        local_bsz = len(batch_ids)
    
        mem_size = sum([self.ops[opname].memory_footprint(local_bsz, ctx_len) for opname in self.ops])
        return mem_size # in bytes
