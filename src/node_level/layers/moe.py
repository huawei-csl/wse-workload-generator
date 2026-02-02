
import logging
import numpy as np 

from src.node_level.layers.linear import Linear
from src.node_level.layers.ffn import FFN
from src.node_level.layers.sum import Sum
from src.node_level.layers.multicast import Multicast
from src.node_level.layers.unicast import Unicast
from src.node_level.layers.barrier import Barrier
from src.node_level.layers.allgather import AllGather

from src.node_level.common.tensor import Tensor, Slice, Concat, get_tensor
from src.node_level.common.compute_graph import reset_compute_graph

from src.node_level.common.workload import get_moe_gate_model
from src.node_level.common.config import SystemConfig
from src.node_level.common.stats import NodeStats
from src.node_level.common.utils import hash_string, dtype_to_byte

class MoE:
    def __init__(self, uid, hidden_size, moe_intermediate_size, num_experts_per_tok, n_experts, n_shared_experts, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating MoE layer {}".format(uid))
        self.uid = uid

        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_experts = n_experts
        self.n_shared_experts = n_shared_experts
        self.dist_info = dist_info
        self.dtype = dtype 
        self.rank = dist_info.rank
        
        if self.dist_info.ep > 1 and dist_info.moe_comm == "allgather":
            self.allgather_dispatch = AllGather(uid+"_ag_disp", hidden_size, dist_info.num_nodes, dist_info, dtype)

        if self.dist_info.ep > 1 and self.dist_info.moe_comm == "allgather":
            self.allgather_combine = AllGather(uid+"_ag_comb", hidden_size, self.dist_info.num_nodes, dist_info, dtype)

        # store which expert resides on which node
        self.expertid_to_node = self.dist_info.get_expert_mapping(self.n_experts)
        local_experts = [expert_id for expert_id in range(self.n_experts) if self.expertid_to_node[expert_id] == self.dist_info.rank_ep]

        self.gate = Linear(uid+"_gate", self.rank, hidden_size, self.n_experts, dtype)

        self.experts = {}
        for i in local_experts:
            self.experts[i] = FFN(uid+"_exp_"+str(i), hidden_size, moe_intermediate_size, dist_info, dtype)
        
        intermediate_size = moe_intermediate_size * n_shared_experts

        self.shared_expert = None
        if self.dist_info.rank in self.dist_info.shared_expert_ranks:
            self.shared_expert = FFN(uid+"_exp_shared", hidden_size, intermediate_size, dist_info, dtype)

    def calc_combine_allgather_traffic(self):
        batch_mappings = self.dist_info.batch_map["attn"]

        expert_routings = get_moe_gate_model().get_expert_routings(layer_id=self.uid)

        outputs = {}
        for expert_id in range(self.n_experts):
            batch_ids = sorted(np.where(expert_routings==expert_id)[1])

            node_id = self.expertid_to_node[expert_id]
            if node_id not in outputs:
                outputs[node_id] = []
            outputs[node_id] += [(expert_id, batch_id) for batch_id in batch_ids]

        for node_id in range(self.dist_info.num_nodes):
            batch_ids = [batch_id for batch_id, shared_expert_node_id in self.dist_info.batch_to_shared_exp.items() if shared_expert_node_id == node_id]
            if len(batch_ids) > 0:
                if node_id not in outputs:
                    outputs[node_id] = []
                outputs[node_id] += [("s", batch_id) for batch_id in batch_ids]


        self.combine_traffic = {}
        for src_id in outputs:
            self.combine_traffic[src_id] = {}
            for dst_id in self.dist_info.ffn_comm_groups["ep"]:
                self.combine_traffic[src_id][dst_id] = list(outputs[src_id])


    def calc_combine_multicast_traffic(self):
        batch_mappings = self.dist_info.batch_map["attn"]

        expert_routings = get_moe_gate_model().get_expert_routings(layer_id=self.uid)

        outputs = {}
        for expert_id in range(self.n_experts):
            batch_ids = sorted(np.where(expert_routings==expert_id)[1])

            node_id = self.expertid_to_node[expert_id]
            if node_id not in outputs:
                outputs[node_id] = []
            outputs[node_id] += [(expert_id, batch_id) for batch_id in batch_ids]

        for node_id in range(self.dist_info.num_nodes):
            batch_ids = [batch_id for batch_id, shared_expert_node_id in self.dist_info.batch_to_shared_exp.items() if shared_expert_node_id == node_id]
            if len(batch_ids) > 0:
                if node_id not in outputs:
                    outputs[node_id] = []
                outputs[node_id] += [("s", batch_id) for batch_id in batch_ids]

        self.combine_traffic = {}
        for node_id in outputs:
            for expert_id, batch_id in outputs[node_id]:
                dp_attn_rank = batch_mappings[batch_id]
                dp_attn_cluster = [k for k,v in self.dist_info.global_cfg.ranks["dp_attn"].items() if v == dp_attn_rank]
                dst = dp_attn_cluster[0]
                if node_id != dst:
                    if node_id not in self.combine_traffic:
                        self.combine_traffic[node_id] = {}

                    if dst not in self.combine_traffic[node_id]:
                        self.combine_traffic[node_id][dst] = []

                    self.combine_traffic[node_id][dst].append((expert_id, batch_id))

    def get_batchids_by_expert(self, expert_id):
        if expert_id == "shared":
            batch_ids = sorted(np.array([batch_id for batch_id, mapped_shared in self.dist_info.batch_to_shared_exp.items() if self.dist_info.rank == mapped_shared]))
            assert len(batch_ids) > 0
        else:
            expert_routings = get_moe_gate_model().get_expert_routings(layer_id=self.uid)
            batch_ids = sorted(np.where(expert_routings==expert_id)[1])
        return batch_ids

    def forward_dispatch_allgather(self, x, stats):
        _, seqlen, hidden_size = x.dims
        
        self.allgather_dispatch.forward(x, stats=stats)

        x_recv = {}

        for e in self.experts:
            recv_batch_ids = self.get_batchids_by_expert(e)
            batch_mappings = self.dist_info.batch_map["attn"]

            recv_buff = []
            for batch_id in recv_batch_ids:
                dp_attn_rank = batch_mappings[batch_id]
                dp_attn_cluster = [k for k,v in self.dist_info.global_cfg.ranks["dp_attn"].items() if v == dp_attn_rank]
                src_id = dp_attn_cluster[0]

                buff_size = sum([v == dp_attn_rank for v in batch_mappings.values()])
                local_buffer = Tensor(f"{x.uid}_ag_{src_id}", self.dist_info.rank, [buff_size, seqlen, hidden_size])

                # offset to substract from batch id to get the correct slice index
                # if bsz=32 and dp_attn=4, each Tensor has a batch size of 8.
                # to slice, say batch id 17, we need to slice 17-16 = 1
                batchid_offset = [v == dp_attn_rank for v in batch_mappings.values()].index(True)

                x_slice = Slice(
                    local_buffer,
                    [batch_id-batchid_offset],
                    axis=0,
                    uid=local_buffer.uid + f"_expert{e}_slice{batch_id}"
                ).forward(stats=stats)
            
                recv_buff.append(x_slice)

            if len(recv_batch_ids) > 0:
                concat_uid = f"{self.uid}_dispatch_recv_concat_expert{e}_" + hash_string("_".join([str(batch_id) for batch_id in recv_batch_ids]))
                x_recv[e] = Concat(
                    recv_buff,
                    axis=0,
                    uid=concat_uid).forward(stats=stats)

        if self.shared_expert:
            shared_recv_buff = []
            for batch_id in self.get_batchids_by_expert("shared"):
                dp_attn_rank = batch_mappings[batch_id]
                dp_attn_cluster = [k for k,v in self.dist_info.global_cfg.ranks["dp_attn"].items() if v == dp_attn_rank]
                src_id = dp_attn_cluster[0]

                batchid_offset = [v == dp_attn_rank for v in batch_mappings.values()].index(True)

                local_buffer = Tensor(f"{x.uid}_ag_{src_id}", self.dist_info.rank, x.dims)
                x_slice = Slice(
                    local_buffer,
                    [batch_id-batchid_offset],
                    axis=0,
                    uid=local_buffer.uid + f"_shared_slice{batch_id}"
                ).forward(stats=stats)
                shared_recv_buff.append(x_slice)

            batch_ids_for_shared = self.get_batchids_by_expert("shared")
            concat_uid = self.uid + "_shared_concat_" + hash_string("_".join([str(batch_id) for batch_id in batch_ids_for_shared]))
            x_recv["shared"] = Concat(
                shared_recv_buff,
                axis=0,
                uid=concat_uid).forward(stats=stats)
                
        return x_recv

    def forward_dispatch_multicast(self, x, stats):
        _, seqlen, hidden_dim = x.dims
        # batch ids processed by this DP cluster
        batch_ids = self.dist_info.get_local_batchids("attn")

        # x has a batch size equal to the number of batch ids processed by this DP cluster. Substract an offset to get the correct slice indices
        minibatch_offset = batch_ids[0]
        for batch_id in batch_ids:
            x_slice = Slice(x, [batch_id-minibatch_offset], axis=0, uid=x.uid + f"_slice{batch_id}").forward(stats=stats)

            # get expert ids for this query
            mapping = get_moe_gate_model().get_mapping_by_batchids(self.uid, batch_id)
            logging.debug("batch_id: {}, mapping: {}".format(batch_id, mapping))

            # after attention allreduce, each node within the DP cluster has identical data
            # therefore, only the master node in the DP cluster dispatches the data to experts
            # TODO: distribute this task to all nodes in the DP cluster for better load balancing
            if self.dist_info.is_dp_master():
                # calculate with nodes the experts are located
                dst_nodes = [self.expertid_to_node[expert_id] for expert_id in mapping.tolist()]

                # Add shared expert node to the destination
                dst_nodes.append(self.dist_info.batch_to_shared_exp[batch_id])

                # remove repeating nodes from dst_nodes
                dst_nodes = list(dict.fromkeys(dst_nodes))

                # remove the nodes from the same DP cluster as they already have the data
                dst_nodes = [node for node in dst_nodes if node not in self.dist_info.dp_attn_cluster]

                # sort the dst_nodes for consistent ordering
                dst_nodes = sorted(dst_nodes)

                if len(dst_nodes) > 0:
                    Multicast(self.uid+"_multicast_exp_"+str(batch_id), dims=x_slice.dims, src=self.dist_info.rank, dst=dst_nodes, dtype=self.dtype).forward(
                        x_slice, stats=stats)

        Barrier(self.uid+"_barrier", nodes=list(range(self.dist_info.num_nodes))).forward(stats=stats) # ensure all nodes have received the multicast before proceeding

        x_recv = {}
        recv_batch_ids = {}
        for e in self.experts:
            recv_batch_ids[e] = self.get_batchids_by_expert(e)
            
            if len(recv_batch_ids[e]) > 0:
                concat_uid = f"{self.uid}_dispatch_recv_concat_expert{e}_" + hash_string("_".join([str(batch_id) for batch_id in recv_batch_ids[e]]))
                x_recv[e] = Concat(
                    [Tensor(f"{x.uid}_slice{batch_id}", self.dist_info.rank, [1, seqlen, hidden_dim]) for batch_id in recv_batch_ids[e]],
                    axis=0,
                    uid=concat_uid).forward(stats=stats)
                
        if self.shared_expert:
            batch_ids_for_shared = self.get_batchids_by_expert("shared")
            concat_uid = self.uid + "_shared_concat_" + hash_string("_".join([str(batch_id) for batch_id in batch_ids_for_shared]))
            x_recv["shared"] = Concat(
                [Tensor(x.uid + f"_slice{batch_id}", self.dist_info.rank, [1, seqlen, hidden_dim]) for batch_id in batch_ids_for_shared],
                axis=0,
                uid=concat_uid).forward(stats=stats)
            
        return x_recv

    def forward_compute_experts(self, x_recv, stats):
        recv_batch_ids = {}
        exp_outs = {}
        total_moe_num_tokens_per_device = 0
        for e in self.experts:
            recv_batch_ids[e] = self.get_batchids_by_expert(e)

            logging.debug("expert {} num of routed samples: {}".format(e, len(recv_batch_ids[e])))

            if len(recv_batch_ids[e]) > 0:
                exp_outs[e] = self.experts[e].forward(x_recv[e], stats=stats)
                
            total_moe_num_tokens_per_device += len(recv_batch_ids[e])

        if self.shared_expert:
            exp_outs["shared"] = self.shared_expert.forward(x_recv['shared'], stats=stats)

        logging.debug("Total number of routed samples for device {}: {}".format(self.dist_info.rank_ep, total_moe_num_tokens_per_device))
        return exp_outs
    
    def forward_combine_allgather(self, exp_outs, stats):
        '''
        Concat all expert outputs in a tensor and perform allgather to gather them
        '''
        if len(exp_outs) == 0:
            return 
        x_local = Concat(
            [exp_outs[e] for e in exp_outs],
            axis=0,
            uid=self.uid+"_combine_concat"
        ).forward(stats=stats)
        
        for dst in self.combine_traffic[self.dist_info.rank]:
            assert x_local.dims[0] == len(self.combine_traffic[self.dist_info.rank][dst])

        out_tensor = self.allgather_combine.forward(x_local, stats=stats)
        return out_tensor

    def forward_combine_multicast(self, exp_outs, stats):
        if len(exp_outs) == 0:
            return 
        
        _, seqlen, hidden_dim = next(iter(exp_outs.values())).dims

        # after MoE layer, gather the outputs in specific nodes to sum them
        # at the moment, we gather them at the dp master of each DP cluster
        # we merge all unicasts to the same dst node
        # TODO: distribute this task to all nodes in the DP cluster for better load balancing
        batchid_dst = {i: [] for i in range(self.dist_info.num_nodes)}

        for e in self.experts:
            # batch ids routed to expert e
            batch_ids = self.get_batchids_by_expert(e)

            # which dp cluster the dst node belongs to
            dp_ranks = self.dist_info.get_dp_rank_from_batchids(batch_ids, "attn")

            # we send the expert outputs to the dp master node of the corresponding dp cluster
            dst_nodes = [self.dist_info.get_dp_master(dp_rank, "attn") for dp_rank in dp_ranks]
            
            for batch_id, dst_node in zip(batch_ids, dst_nodes):
                batchid_dst[dst_node].append((batch_id, e))

        # sort the batchids for each dst_node, first by the expert_id then by batch_id
        for dst_node in batchid_dst:
            batchid_dst[dst_node] = sorted(batchid_dst[dst_node], key=lambda x: (x[1], x[0]) ) # sort by batch_id

        if self.shared_expert:
            # batch ids routed to shared expert
            # batch_ids = [batch_id for batch_id, mapped_shared in self.dist_info.batch_to_shared_exp.items() if self.dist_info.rank == mapped_shared]
            batch_ids = self.get_batchids_by_expert("shared")

            # which dp cluster the dst node belongs to
            dp_ranks = self.dist_info.get_dp_rank_from_batchids(batch_ids, "attn")

            # we send the expert outputs to the dp master node of the corresponding dp cluster
            dst_nodes = [self.dist_info.get_dp_master(dp_rank, "attn") for dp_rank in dp_ranks]
            
            for batch_id, dst_node in zip(batch_ids, dst_nodes):
                batchid_dst[dst_node].append((batch_id, "shared"))

        for dst_node in batchid_dst:
            # if dst node is itself, skip
            if dst_node == self.dist_info.rank:
                continue

            # if there are tokens to send to node i, do unicast
            if len(batchid_dst[dst_node]) > 0:
                exp_outs_to_send = [] 
                for batch_id, expert_id in batchid_dst[dst_node]:
                    if expert_id == "shared":
                        exp_input_batch_ids = self.get_batchids_by_expert("shared")

                        minibatch_ind = np.argwhere(exp_input_batch_ids == batch_id).item()
                        exp_outs_to_send.append(
                            Slice(exp_outs[expert_id], [minibatch_ind], axis=0, uid=exp_outs[expert_id].uid + f"_slice{batch_id}").forward(stats=stats)
                        )
                    else:
                        exp_input_batch_ids = self.get_batchids_by_expert(expert_id)

                        minibatch_ind = np.argwhere(exp_input_batch_ids == batch_id).item()
                        exp_outs_to_send.append(
                            Slice(exp_outs[expert_id], [minibatch_ind], axis=0, uid=exp_outs[expert_id].uid + f"_slice{batch_id}").forward(stats=stats)
                        )
                
                concat_uid = self.uid + "_gather_unicast_concat_" + hash_string("_".join([f"{batch_id}e{expert_id}" for batch_id, expert_id in batchid_dst[dst_node]]))
                unicast_tensor = Concat(
                    exp_outs_to_send, 
                    axis=0,
                    uid=concat_uid).forward(stats=stats)
                Unicast(self.uid+"_unicast_"+str(dst_node), dims=unicast_tensor.dims, src=self.dist_info.rank, dst=dst_node, dtype=self.dtype).forward(
                    unicast_tensor, stats=stats)
                
                assert len(batchid_dst[dst_node]) == unicast_tensor.dims[0]

        Barrier(self.uid+"_barrier_uc", nodes=list(range(self.dist_info.num_nodes))).forward(stats=stats) # ensure all nodes have received the unicast before proceeding


    def forward_combine_allgather_add(self, exp_outs, ag_out, stats):
        _, seqlen, hidden_dim = next(iter(exp_outs.values())).dims

        batch_ids = self.dist_info.get_local_batchids("attn")

        out_sums = []
        for batch_id in batch_ids:
            mapping = get_moe_gate_model().get_mapping_by_batchids(self.uid, batch_id)
            expert_ids = [expert_id for expert_id in mapping.tolist()]

            exp_outs_to_recv = []
            for expert_id in expert_ids:
                src_id = self.expertid_to_node[expert_id]

                exp_input_batch_ids = self.get_batchids_by_expert(expert_id)
                buff_ind = exp_input_batch_ids.index(batch_id)
                # uid = f"{exp_outs[expert_id].uid}_node{src_id}_slice{buff_ind}"

                recv_buffer = Tensor(f"{self.uid}_combine_concat_ag_{src_id}", self.dist_info.rank, [len(exp_input_batch_ids), seqlen, hidden_dim])

                exp_outs_to_recv.append(
                    Slice(recv_buffer, [buff_ind], axis=0, uid=recv_buffer.uid + f"_expert{expert_id}_slice{buff_ind}").forward(stats=stats)
                )

            src_id = self.dist_info.batch_to_shared_exp[batch_id]
            recv_buffer = Tensor(f"{self.uid}_combine_concat_ag_{src_id}", self.dist_info.rank, [len(self.combine_traffic[src_id][self.dist_info.rank]), seqlen, hidden_dim])

            buff_offset = [v == src_id for v in self.dist_info.batch_to_shared_exp.values()].index(True)
            # uid = f"{exp_outs['shared'].uid}_node{src_id}_slice{buff_ind}"
            exp_outs_to_recv.append(
                Slice(recv_buffer, [batch_id-buff_offset], axis=0, uid=recv_buffer.uid + f"_shared_slice{batch_id-buff_offset}").forward(stats=stats)
            )

            assert len(exp_outs_to_recv) == self.num_experts_per_tok + self.n_shared_experts

            concat_uid = f"{self.uid}_combine_add_concat_batchid{batch_id}"

            x_recv = Concat(
                    exp_outs_to_recv, 
                    axis=0,
                    uid=concat_uid).forward(stats=stats)

            out_sums.append(
                Sum(self.uid+"_sum_"+str(batch_id), dims=x_recv.dims, axis=0, dist_info=self.dist_info, dtype=self.dtype).forward(x_recv, stats=stats)
            ) 

        out_batch = Concat(
            out_sums,
            axis=0,
            uid=self.uid+"_combine_out_concat").forward(stats=stats)

        assert out_batch.dims[0] == len(batch_ids)

        return out_batch

    def forward_combine_multicast_add(self, exp_outs, stats):
        _, seqlen, hidden_dim = next(iter(exp_outs.values())).dims

        batch_ids = self.dist_info.get_local_batchids("attn")

        recv_buffer = {}
        for src_id in self.combine_traffic:
            if self.dist_info.rank in self.combine_traffic[src_id]:
                recv_buffer[src_id] = Tensor(f"{self.uid}_unicast_{self.dist_info.rank}_{src_id}", self.dist_info.rank, [len(self.combine_traffic[src_id][self.dist_info.rank]), seqlen, hidden_dim])
        
        out_sums = []
        for batch_id in batch_ids:
            mapping = get_moe_gate_model().get_mapping_by_batchids(self.uid, batch_id)
            expert_ids = [expert_id for expert_id in mapping.tolist()]
            src_ids = [self.expertid_to_node[expert_id] for expert_id in mapping.tolist()]

            exp_outs_to_recv = []
            for expert_id in expert_ids:
                src_id = self.expertid_to_node[expert_id]

                if src_id == self.dist_info.rank:
                    exp_input_batch_ids = self.get_batchids_by_expert(expert_id)
                    buff_ind = exp_input_batch_ids.index(batch_id)
                    uid = f"{exp_outs[expert_id].uid}_node{src_id}_slice{buff_ind}"
                    exp_outs_to_recv.append(
                        Slice(exp_outs[expert_id], [buff_ind], axis=0, uid=uid).forward(stats=stats)
                    )
                else:
                    buff_ind = self.combine_traffic[src_id][self.dist_info.rank].index((expert_id, batch_id))
                    exp_outs_to_recv.append(
                        Slice(recv_buffer[src_id], [buff_ind], axis=0, uid=recv_buffer[src_id].uid + f"_slice{buff_ind}").forward(stats=stats)
                    )
            
            src_id = self.dist_info.batch_to_shared_exp[batch_id]
            if src_id == self.dist_info.rank:
                exp_input_batch_ids = self.get_batchids_by_expert("shared")
                buff_ind = exp_input_batch_ids.index(batch_id)
                uid = f"{exp_outs['shared'].uid}_node{src_id}_slice{buff_ind}"
                exp_outs_to_recv.append(
                    Slice(exp_outs["shared"], [buff_ind], axis=0, uid=uid).forward(stats=stats)
                )
            else:
                buff_ind = self.combine_traffic[src_id][self.dist_info.rank].index(("s", batch_id))
                exp_outs_to_recv.append(
                    Slice(recv_buffer[src_id], [buff_ind], axis=0, uid=recv_buffer[src_id].uid + f"_slice{buff_ind}").forward(stats=stats)
                )

            concat_uid = f"{self.uid}_combine_add_concat_batchid{batch_id}"

            assert len(exp_outs_to_recv) == self.num_experts_per_tok + self.n_shared_experts

            x_recv = Concat(
                    exp_outs_to_recv, 
                    axis=0,
                    uid=concat_uid).forward(stats=stats)

            out_sums.append(
                Sum(self.uid+"_sum_"+str(batch_id), dims=x_recv.dims, axis=0, dist_info=self.dist_info, dtype=self.dtype).forward(x_recv, stats=stats)
            ) 

        out_batch = Concat(
            out_sums,
            axis=0,
            uid=self.uid+"_combine_out_concat").forward(stats=stats)

        return out_batch

    def forward_distribute_dp(self, x, stats):
        # once all unicasts are done, perform a multicast within the DP cluster for the next layer
        batch_ids = self.dist_info.get_local_batchids("attn")
        multicast_dsts = [dst for dst in self.dist_info.dp_attn_cluster if dst != self.dist_info.rank]
        Multicast(self.uid+"_multicast_dp", dims=x.dims, src=self.dist_info.rank, dst=multicast_dsts, dtype=self.dtype).forward(
            x, stats=stats)
    
        # Barrier(self.uid+"_barrier_mc", nodes=self.dist_info.dp_attn_cluster).forward(stats=stats) # ensure all nodes in the DP cluster have received the multicast before proceeding

    def forward(self, x, stats): 
        self.global_bsz = get_moe_gate_model().global_bsz
        
        # perform MoE gating
        self.gate.forward(x, stats)

        # dispatch the inputs to other nodes based on expert routing
        if self.dist_info.ep > 1:
            if self.dist_info.moe_comm == "allgather":
                expert_recv_buffs = self.forward_dispatch_allgather(x, stats=stats)
            elif self.dist_info.moe_comm == "multicast":     
                expert_recv_buffs = self.forward_dispatch_multicast(x, stats=stats)
            else:
                raise NotImplementedError("MoE communication method {} not implemented".format(self.dist_info.moe_comm))

        # compute the outputs for each local expert
        exp_outs = self.forward_compute_experts(expert_recv_buffs, stats=stats)

        # combine the outputs from all experts
        if self.dist_info.ep > 1:
            if self.dist_info.moe_comm == "allgather":
                self.calc_combine_allgather_traffic()
                ag_out = self.forward_combine_allgather(exp_outs, stats=stats)
                out_tensor = self.forward_combine_allgather_add(exp_outs, ag_out, stats=stats)

            elif self.dist_info.moe_comm == "multicast":
                self.calc_combine_multicast_traffic()

                # first gather all expert outputs at dp masters
                self.forward_combine_multicast(exp_outs, stats=stats)

                if self.dist_info.is_dp_master():
                    out_tensor = self.forward_combine_multicast_add(exp_outs, stats=stats)

                    # then distribute the outputs within each DP cluster
                    self.forward_distribute_dp(out_tensor, stats=stats)
                else:
                    out_tensor = Tensor(
                        f"{self.uid}_multicast_dp", 
                        self.dist_info.rank,
                        dims=x.dims)

                Barrier(self.uid+"_barrier_mc", nodes=self.dist_info.dp_attn_cluster).forward(stats=stats) # ensure all nodes in the DP cluster have received the multicast before proceeding
            else:
                raise NotImplementedError("MoE communication method {} not implemented".format(self.dist_info.moe_comm))
        else:
            out_tensor = Tensor(
                f"{self.uid}_multicast_dp", 
                self.dist_info.rank,
                dims=x.dims)
        
        return out_tensor

    def memory_footprint(self, bsz=None, ctx_len=None):
        mem_size = self.gate.memory_footprint()
        mem_size += sum([self.experts[e].memory_footprint() for e in self.experts])
        if self.shared_expert:
            mem_size += self.shared_expert.memory_footprint()
        return mem_size # in bytes

    def routings_summary(self):
        global_bsz = get_moe_gate_model().global_bsz

        # A dict that store the batch id to expert id mapping for each DP cluster
        # key: batch id, value: dp attn rank
        batch_mappings = self.dist_info.batch_map["attn"]

        # A 2-D array that store the expert routing for each token
        # shape: (num_experts_per_token, global_bsz)
        expert_routings = get_moe_gate_model().get_expert_routings(layer_id=self.uid)

        logging.info("\n---- Mapping ----\n")

        dispatch_traffic = [[[] for j in range(self.dist_info.num_nodes)] for i in range(self.dist_info.num_nodes)]
        for batch_id in range(global_bsz):
            mapped_to_expert_ids = expert_routings[:, batch_id]
            mapped_to_nodes = [self.expertid_to_node[e] for e in mapped_to_expert_ids]
            
            # shared expert node
            shared_expert_node_id = self.dist_info.batch_to_shared_exp[batch_id]
            mapped_to_nodes.append(shared_expert_node_id)

            dp_attn_rank = batch_mappings[batch_id]
            dp_attn_cluster = [k for k,v in self.dist_info.global_cfg.ranks["dp_attn"].items() if v == dp_attn_rank]
            send_to = sorted(list(dict.fromkeys([node_id for node_id in mapped_to_nodes if node_id not in dp_attn_cluster])))

            logging.info(f"Sample {batch_id} is mapped to expert {mapped_to_expert_ids}. These experts reside in nodes {mapped_to_nodes}. Nearest shared expert reside in {shared_expert_node_id}. Sample already exists in nodes {dp_attn_cluster}. It should be sent to {send_to}")

            src = dp_attn_cluster[0]
            for dst in send_to:
                dispatch_traffic[src][dst].append(batch_id)

        logging.info("\n---- Dispatch Send ----\n")

        for src_id in range(self.dist_info.num_nodes):
            for dst_id in range(self.dist_info.num_nodes):
                if len(dispatch_traffic[src_id][dst_id]) > 0:
                    print(f"Node {src_id} sends {len(dispatch_traffic[src_id][dst_id])} samples to node {dst_id}: {dispatch_traffic[src_id][dst_id]}")

        logging.info("\n---- Dispatch Receive ----\n")

        for dst_id in range(self.dist_info.num_nodes):
            for src_id in range(self.dist_info.num_nodes):
                if len(dispatch_traffic[src_id][dst_id]) > 0:
                    print(f"Node {dst_id} receives {len(dispatch_traffic[src_id][dst_id])} samples from node {src_id}: {dispatch_traffic[src_id][dst_id]}")

        logging.info("\n---- Expert Processing ----\n")

        outputs = {}
        for expert_id in range(self.n_experts):
            batch_ids = sorted(np.where(expert_routings==expert_id)[1])
            logging.info(f"Expert {expert_id} on node {self.expertid_to_node[expert_id]} processes {len(batch_ids)} samples: {batch_ids}")

            node_id = self.expertid_to_node[expert_id]
            if node_id not in outputs:
                outputs[node_id] = []
            outputs[node_id] += [(expert_id, batch_id) for batch_id in batch_ids]

        for node_id in range(self.dist_info.num_nodes):
            batch_ids = [batch_id for batch_id, shared_expert_node_id in self.dist_info.batch_to_shared_exp.items() if shared_expert_node_id == node_id]
            if len(batch_ids) > 0:
                logging.info(f"Shared expert on node {node_id} processes {len(batch_ids)} samples: {batch_ids}")

                if node_id not in outputs:
                    outputs[node_id] = []
                outputs[node_id] += [("s", batch_id) for batch_id in batch_ids]
        
        for node_id in outputs:
            output_ids = ", ".join(f"{expert_id}_{batch_id}" for expert_id, batch_id in outputs[node_id])
            logging.info(f"Node {node_id} produced outputs (e_s): {output_ids}")

        logging.info("\n---- Combine Traffic Send ----\n")
        combine_traffic = [[[] for j in range(self.dist_info.num_nodes)] for i in range(self.dist_info.num_nodes)]
        for node_id in outputs:
            for expert_id, batch_id in outputs[node_id]:
                dp_attn_rank = batch_mappings[batch_id]
                dp_attn_cluster = [k for k,v in self.dist_info.global_cfg.ranks["dp_attn"].items() if v == dp_attn_rank]
                dst = dp_attn_cluster[0]
                if node_id != dst:
                    logging.info(f"Node {node_id} sends output {expert_id}_{batch_id} back to DP cluster {dp_attn_cluster[0]}")
                    combine_traffic[node_id][dst].append((expert_id, batch_id))

        for src_id in range(self.dist_info.num_nodes):
            for dst_id in range(self.dist_info.num_nodes):
                if len(combine_traffic[src_id][dst_id]) > 0:
                    output_ids = ", ".join([f"{expert_id}_{batch_id}" for expert_id, batch_id in combine_traffic[src_id][dst_id]])
                    logging.info(f"Node {src_id} sends {len(combine_traffic[src_id][dst_id])} outputs to node {dst_id}: {output_ids}")

        logging.info("\n---- Combine Traffic Receive ----\n")
        for dst_id in range(self.dist_info.num_nodes):
            for src_id in range(self.dist_info.num_nodes):
                if len(combine_traffic[src_id][dst_id]) > 0:
                    output_ids = ", ".join([f"{expert_id}_{batch_id}" for expert_id, batch_id in combine_traffic[src_id][dst_id]])
                    logging.info(f"Node {dst_id} receives {len(combine_traffic[src_id][dst_id])} outputs from node {src_id}: {output_ids}")        

        logging.info("\n---- Weighted Sum ----\n")
        for node_id in range(self.dist_info.num_nodes):
            dp_attn_rank = self.dist_info.global_cfg.ranks["dp_attn"][node_id]
            dp_attn_cluster = [k for k,v in self.dist_info.global_cfg.ranks["dp_attn"].items() if v == dp_attn_rank]
            if node_id == dp_attn_cluster[0]:
                batch_ids = [batch_id for batch_id in range(global_bsz) if self.dist_info.batch_map["attn"][batch_id] == dp_attn_rank]

                for batch_id in batch_ids:
                    mapped_to_expert_ids = expert_routings[:, batch_id]
                    logging.info(f"Node {node_id} performs weighted sum for sample {batch_id} with expert outputs: ", [f"{expert_id}_{batch_id}" for expert_id in mapped_to_expert_ids])

        logging.info("\n---- Distribute within DP cluster ----\n")
        for node_id in range(self.dist_info.num_nodes):
            dp_attn_rank = self.dist_info.global_cfg.ranks["dp_attn"][node_id]
            dp_attn_cluster = [k for k,v in self.dist_info.global_cfg.ranks["dp_attn"].items() if v == dp_attn_rank]
            if node_id == dp_attn_cluster[0]:
                batch_ids = [batch_id for batch_id in range(global_bsz) if self.dist_info.batch_map["attn"][batch_id] == dp_attn_rank]

                for batch_id in batch_ids:
                    logging.info(
                        f"Node {node_id} sends output for sample {batch_id} in the same DP cluster to nodes:"
                        " ".join([str(nid) for nid in dp_attn_cluster if nid != node_id])
                    )
        
        return dispatch_traffic, combine_traffic

if __name__ == "__main__":
    bsz = 4
    seqlen = 1

    hidden_size = 7168
    moe_intermediate_size = 2048
    num_experts_per_tok = 8
    n_experts = 16
    n_shared_experts = 1
    n_redundant_shared_exp = 1
    dtype = "fp16"

    ep = 8
    num_nodes = ep

    dp_attn = 1
    tp_attn = num_nodes // dp_attn

    for rank in [0, 1, 2, 3, 4, 5, 6, 7]:
        reset_compute_graph()

        stats = NodeStats()
        stats.new_iter(iter_id=0)

        dist_info = SystemConfig().from_args(
                num_nodes=num_nodes,
                dp_attn=dp_attn,
                tp_attn=tp_attn,
                ep=ep,
                moe_comm="multicast",
                expert_workload_model="uniform",
                n_redundant_shared_exp=n_redundant_shared_exp,
        ).get_dist_info(rank)
        dist_info.batch_mapping(bsz)

        moe_gate_model = get_moe_gate_model(num_experts_per_tok, n_experts, ["moe_0"], dist_info.expert_workload_model)
        moe_gate_model.new_iter(iter_id=0, bsz=bsz, seqlen=seqlen)

        moe_layer = MoE(
            uid=f"moe_0",
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            n_experts=n_experts,
            n_shared_experts=n_shared_experts,
            dist_info=dist_info,
            dtype=dtype
        )

        batch_ids = dist_info.get_local_batchids("attn")
        local_bsz = len(batch_ids)

        x = Tensor("input", dist_info.rank, [bsz, seqlen, hidden_size])
        x_local = Slice(x, [batch_id for batch_id in batch_ids], axis=0).forward(stats=stats)

        y = moe_layer.forward(x_local, stats=stats)

        assert y.dims == [local_bsz, seqlen, hidden_size], f"Output dims {y.dims} do not match expected dims {[local_bsz, seqlen, hidden_size]}"

        op_mem_foot, op_num_ops, op_hbm_reads, op_net_data = stats.sumUp()

        expert_routing = moe_gate_model.get_expert_routings(layer_id="moe_0")

        local_experts = []
        for expert_id, rank_ep in dist_info.get_expert_mapping(n_experts).items():
            if rank == rank_ep:
                local_experts.append(expert_id)

        mem_footprint_per_expert = 3 * hidden_size * moe_intermediate_size * dtype_to_byte(dtype) # weights only
        
        expected_footprint = hidden_size * n_experts * dtype_to_byte(dtype) # gate weights
        expected_footprint += len(local_experts) * mem_footprint_per_expert # local routed experts
        # if this node holds a shared expert
        if rank in dist_info.shared_expert_ranks:
            expected_footprint += mem_footprint_per_expert

        expected_num_ops = local_bsz * seqlen * hidden_size * n_experts # gating
        expected_hbm_reads = hidden_size * n_experts * dtype_to_byte(dtype) # gating
        for expert_id in local_experts:
            num_tokens = len(np.where(expert_routing==expert_id)[1])
            expected_num_ops += 3 * num_tokens * hidden_size * moe_intermediate_size
            if num_tokens > 0:
                expected_hbm_reads += 3 * hidden_size * moe_intermediate_size * dtype_to_byte(dtype) # weights

        batch_ids_for_shared = [batch_id for batch_id, mapped_shared in dist_info.batch_to_shared_exp.items() if dist_info.rank == mapped_shared]
        if rank in dist_info.shared_expert_ranks:
            expected_num_ops += 3 * len(batch_ids_for_shared) * hidden_size * (moe_intermediate_size * n_shared_experts)
            expected_hbm_reads += 3 * hidden_size * (moe_intermediate_size * n_shared_experts) * dtype_to_byte(dtype) # weights

        dispatch_traffic, combine_traffic = moe_layer.routings_summary()

        expected_network_data = 0
        src_id = rank
        for dst_id in range(len(dispatch_traffic)):
            expected_network_data += len(dispatch_traffic[src_id][dst_id]) * seqlen * hidden_size * dtype_to_byte(dtype)

        for dst_id in range(len(combine_traffic)):
            expected_network_data += len(combine_traffic[src_id][dst_id]) * seqlen * hidden_size * dtype_to_byte(dtype)

        # DP Cluster distribute multicast traffic
        dp_attn_cluster_size = len(dist_info.dp_attn_cluster)
        if dist_info.is_dp_master():
            expected_network_data += local_bsz * seqlen * hidden_size * dtype_to_byte(dtype) * (dp_attn_cluster_size-1)

        assert expected_num_ops == op_num_ops, f"Expected num_ops {expected_num_ops}, got {op_num_ops}"
        assert expected_hbm_reads == op_hbm_reads, f"Expected hbm_reads {expected_hbm_reads}, got {op_hbm_reads}"
        assert expected_network_data == op_net_data, f"Expected network data {expected_network_data}, got {op_net_data}"