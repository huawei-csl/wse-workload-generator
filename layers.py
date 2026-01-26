import logging
from typing import List

import node_level.common.compute_graph as compute_graph
from node_level.common.compute_graph import get_compute_graph

from node_level.layers.linear import Linear
from node_level.layers.allreduce import Allreduce
from node_level.layers.mla_absorb_block import MLAAbsorbBlock
from node_level.layers.ffn import FFN

from node_level.common.tensor import Tensor, get_tensor, View, Split, Concat, Slice, Transpose
from utils import dtype_to_byte, intceil, divide_equal, hash_string
from workload import get_moe_gate_model

class Layer:
    def __init__(self) -> None:
        pass

    def forward(self, bsz=None, seqlen=None, ctx_len=None, stats=None):
        raise NotImplementedError

    def memory_footprint(self, bsz=None, ctx_len=None):
        raise NotImplementedError

    def num_ops(self, bsz=None, ctx_len=None):
        raise NotImplementedError
    
    def hbm_reads(self, bsz=None, ctx_len=None):
        raise NotImplementedError
    
    def network_data(self, bsz=None, ctx_len=None):
        raise NotImplementedError    

class Sum(Layer):
    def __init__(self, uid, dims, axis, dist_info, dtype) -> None:
        super().__init__()
        logging.debug("Sum layer {} with dims: {}".format(uid, dims))

        self.uid = uid 
        self.dims = dims
        self.axis = axis

        self.out_dims = list(self.dims)
        self.out_dims[self.axis] = 1

        self.dist_info = dist_info
        self.dtype = dtype

    def forward(self, x, stats=None):
        assert self.dims == x.dims, "Input dims {} does not match layer dims {}".format(x.dims, self.dims)

        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims()

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))

        out = Tensor(f"{self.uid}_out", self.dist_info.rank, self.out_dims) # squeeze seqlen
        stats.append(self.uid, "Sum", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0
    
    def get_dims(self):
        return str(self.dims) + " -> " + str(self.axis) + " -> " + str(self.out_dims)
    
    def num_ops(self):
        # n_ops = eval("*".join([str(d) for d in self.dims]))
        n_ops = 0
        return n_ops # in terms of number of MACs

    def hbm_reads(self):
        # rw = eval("*".join([str(d) for d in self.dims])) * dtype_to_byte(self.dtype)
        rw = 0
        return rw # weights only, in bytes

    def network_data(self):
        return 0

class AlltoAll(Layer):
    def __init__(self, uid, vector_size, cluster_size, dist_info, dtype) -> None:
        super().__init__()
        logging.debug("AlltoAll layer {} with vector size: {} among {} devices".format(uid, vector_size, cluster_size))

        self.uid = uid
        self.vector_size = vector_size
        self.cluster_size = cluster_size
        self.dtype = dtype
        self.dist_info = dist_info
        self.comm_group = list(range(cluster_size))

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data(bsz*seqlen)
        dims = self.get_dims(bsz*seqlen)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        out = Tensor(f"{self.uid}_out", self.dist_info.rank, (bsz, seqlen, hidden_dim))
        stats.append(self.uid, "AlltoAll", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.comm_group, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)
        return out 
    
    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def get_dims(self, bsz):
        vec_dims = [bsz, self.vector_size]
        return str(vec_dims)
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self, bsz):
        vecsize = 2 * bsz * self.vector_size * (self.cluster_size - 1) * dtype_to_byte(self.dtype) # N-1 vec receive + N-1 vec send, N: no. of devices in a cluster
        logging.debug("{}: network data size (send + receive): {} B".format(self.uid, vecsize))
        return vecsize # in bytes


class Unicast(Layer):
    def __init__(self, uid, dims, src: int, dst: int, dtype) -> None:
        super().__init__()
        logging.debug("Unicast layer {} with vector size: {} src:{} dst:{}".format(uid, dims, src, dst))

        self.uid = uid
        self.dims = dims
        self.dtype = dtype
        self.src = src
        self.dst = dst

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        assert x.dims == self.dims, "Input vector size {} does not match expected vector size {}".format(x.dims, self.dims)
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims()

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        
        out = Tensor(f"{self.uid}_{self.src}", self.dst, (bsz, seqlen, hidden_dim))
        stats.append(self.uid, "Unicast", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.dst, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def get_dims(self):
        return self.dims
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self):
        vecsize = eval("*".join([str(d) for d in self.dims])) * dtype_to_byte(self.dtype) # a vec of this size is sent from a single source to multiple destionations
        logging.debug("{}: network data size: {} B".format(self.uid, vecsize))
        return vecsize # in bytes

class Multicast(Layer):
    def __init__(self, uid, dims, src: int, dst: List[int], dtype) -> None:
        super().__init__()
        logging.debug("Multicast layer {} with dims: {} src:{} dst:{}".format(uid, dims, src, dst))

        self.uid = uid
        self.dims = dims
        self.dtype = dtype
        self.src = src
        self.dst = dst

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        assert x.dims == self.dims, "Input dims {} do not match expected dims {}".format(x.dims, self.dims)
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims()

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        
        out = [Tensor(f"{x.uid}", d, (bsz, seqlen, hidden_dim)) for d in self.dst]
        stats.append(self.uid, "Multicast", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.dst, dims=dims)
        get_compute_graph().add_node(self, [x], out, attrs=None)

        return out 
    
    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def get_dims(self):
        return self.dims
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self):
        vector_size = eval("*".join([str(d) for d in self.dims])) * len(self.dst) * dtype_to_byte(self.dtype) # a vec of this size is sent from a single source to multiple destinations
        logging.debug("{}: network data size: {} B".format(self.uid, vector_size))
        return vector_size # in bytes

class Barrier(Layer):
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

class SelfAttention(Layer):
    def __init__(self, uid, num_attention_heads, num_key_value_heads, head_dim, seq_parallel, dist_info, dtype) -> None:
        super().__init__()
        logging.debug("SelfAttention layer {} with KV-cache dims: bsz x ctx_len x {} x {}".format(uid, num_key_value_heads, head_dim))

        self.uid = uid
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.seq_parallel = seq_parallel
        self.dist_info = dist_info

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)
        dims = self.get_dims(bsz, seqlen, ctx_len)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        stats.append(self.uid, "SelfAttention", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        
    def memory_footprint(self, bsz, ctx_len):
        memory_footprint = 2 * bsz * intceil(ctx_len/self.seq_parallel) * self.num_key_value_heads * self.head_dim * dtype_to_byte(self.dtype) # KV-cache
        return memory_footprint  # KV-cache only, in bytes

    def get_dims(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.seq_parallel)
            input_dims = [bsz, seqlen, self.n_local_heads, self.qk_head_dim]
            K_dims = [bsz, seqlen_per_device, self.n_local_heads, self.qk_head_dim]
            V_dims = [bsz, seqlen_per_device, self.n_local_heads, self.v_head_dim]
            out_dims = [bsz, seqlen, self.n_local_heads, self.v_head_dim]
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            input_dims = [bsz, 1, self.n_local_heads, self.qk_head_dim]
            K_dims = [bsz, ctx_len_per_device, self.n_local_heads, self.qk_head_dim]
            V_dims = [bsz, ctx_len_per_device, self.n_local_heads, self.v_head_dim]
            out_dims = [bsz, 1, self.n_local_heads, self.v_head_dim]
        return ",".join(input_dims) + " x " + ",".join(K_dims) + " x " + ",".join(V_dims) + " -> " + ",".join(out_dims)
    
    def get_dims(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.seq_parallel)
            input_dims = [bsz, seqlen, self.num_attention_heads, self.head_dim]
            K_dims = [bsz, seqlen_per_device, self.num_key_value_heads, self.head_dim]
            V_dims = [bsz, seqlen_per_device, self.num_key_value_heads, self.head_dim]
            out_dims = [bsz, seqlen, self.num_attention_heads, self.head_dim]
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            input_dims = [bsz, 1, self.num_attention_heads, self.head_dim]
            K_dims = [bsz, ctx_len_per_device, self.num_key_value_heads, self.head_dim]
            V_dims = [bsz, ctx_len_per_device, self.num_key_value_heads, self.head_dim]
            out_dims = [bsz, 1, self.num_attention_heads, self.head_dim]
        return "Q: " + str(input_dims) + ", K: " + str(K_dims) + ", V: " + str(V_dims) + " -> " + str(out_dims)
    
    def num_ops(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            n_ops = bsz * intceil(seqlen/self.seq_parallel) * self.num_attention_heads * self.head_dim * seqlen # QKT
            n_ops += bsz * intceil(seqlen/self.seq_parallel) * self.num_attention_heads * self.head_dim * seqlen # SV
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            logging.debug("{} bsz: {}, ctx_len: {}, num_attention_heads: {}, head_dim: {}, seqlen: {}".format(self.uid, bsz, ctx_len_per_device, self.num_attention_heads, self.head_dim, seqlen))
            n_ops = bsz * ctx_len_per_device * self.num_attention_heads * self.head_dim * seqlen # QKT
            n_ops += bsz * ctx_len_per_device * self.num_attention_heads * self.head_dim * seqlen # SV
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
        logging.debug("{} bsz: {}, ctx_len: {}, num_attention_heads: {}, head_dim: {}".format(self.uid, bsz, ctx_len_per_device, self.num_attention_heads, self.head_dim))
        rw = 2 * bsz * ctx_len_per_device * self.num_key_value_heads * self.head_dim * dtype_to_byte(self.dtype) # KV-cache
        return rw # KV-cache only, in bytes

    def network_data(self, bsz=None):
        return 0

class MLANaiveAttention(Layer):
    def __init__(self, uid, n_local_heads, qk_head_dim, v_head_dim, dist_info, dtype) -> None:
        super().__init__()

        self.uid = uid 
        self.n_local_heads = n_local_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.dtype = dtype
        self.dist_info = dist_info

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)
        dims = self.get_dims(bsz, seqlen, ctx_len)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        stats.append(self.uid, "MLANaiveAttention", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)

    def memory_footprint(self, bsz, ctx_len):
        memory_footprint = bsz * intceil(ctx_len/self.dist_info.sp) * self.n_local_heads * self.qk_head_dim * dtype_to_byte(self.dtype) # k_cache
        memory_footprint += bsz * intceil(ctx_len/self.dist_info.sp) * self.n_local_heads * self.v_head_dim * dtype_to_byte(self.dtype) # v_cache
        return memory_footprint  # KV-cache only, in bytes

    def get_dims(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.dist_info.sp)
            input_dims = [bsz, seqlen, self.n_local_heads, self.qk_head_dim]
            K_dims = [bsz, seqlen_per_device, self.n_local_heads, self.qk_head_dim]
            V_dims = [bsz, seqlen_per_device, self.n_local_heads, self.v_head_dim]
            out_dims = [bsz, seqlen, self.n_local_heads, self.v_head_dim]
        else:
            ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
            input_dims = [bsz, 1, self.n_local_heads, self.qk_head_dim]
            K_dims = [bsz, ctx_len_per_device, self.n_local_heads, self.qk_head_dim]
            V_dims = [bsz, ctx_len_per_device, self.n_local_heads, self.v_head_dim]
            out_dims = [bsz, 1, self.n_local_heads, self.v_head_dim]
        return "Q: " + str(input_dims) + ", K: " + str(K_dims) + ", V: " + str(V_dims) + " -> " + str(out_dims)
    
    def num_ops(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.dist_info.sp)
            n_ops = bsz * seqlen_per_device * self.n_local_heads * self.qk_head_dim * seqlen # einsum(bshd,bthd→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.v_head_dim * seqlen # einsum(bsht,bthv→bshv)
        else:
            ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
            n_ops = bsz * ctx_len_per_device * self.n_local_heads * self.qk_head_dim * seqlen # einsum(bshd,bthd→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.v_head_dim * seqlen # einsum(bsht,bthv→bshv)
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
        rw = bsz * ctx_len_per_device * self.n_local_heads * self.qk_head_dim * dtype_to_byte(self.dtype) # k_cache
        rw += bsz * ctx_len_per_device * self.n_local_heads * self.v_head_dim * dtype_to_byte(self.dtype) # v_cache
        return rw # KV-cache only, in bytes

    def network_data(self, bsz=None):
        return 0
    
class GQABlock(Layer):
    def __init__(self, uid, hidden_size, num_attention_heads, num_key_value_heads, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating GQA block {}".format(uid))

        assert hidden_size % num_attention_heads == 0
        head_dim = hidden_size // num_attention_heads

        self.dist_info = dist_info

        num_heads_per_device = intceil(num_attention_heads / dist_info.tp_attn)
        num_kv_heads_per_device = intceil(num_key_value_heads / dist_info.tp_attn)

        self.uid = uid
        self.rank = dist_info.rank

        self.ops = {}
        self.ops["q_proj"] = Linear(uid+"_qproj", self.rank, hidden_size, num_heads_per_device * head_dim, dtype)
        self.ops["k_proj"] = Linear(uid+"_kproj", self.rank, hidden_size, num_kv_heads_per_device * head_dim, dtype)
        self.ops["v_proj"] = Linear(uid+"_vproj", self.rank, hidden_size, num_kv_heads_per_device * head_dim, dtype)

        self.ops["self_attn"] = SelfAttention(uid+"_selfattn", num_heads_per_device, num_kv_heads_per_device, head_dim, dist_info.sp, dtype)
        if dist_info.sp > 1:
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", self.rank, num_heads_per_device * head_dim, dist_info.attn_comm_groups["sp"], dtype)

        self.ops["o_proj"] = Linear(uid+"_oproj", self.rank, num_heads_per_device * head_dim, hidden_size, dtype)

        if dist_info.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", self.rank, hidden_size, dist_info.attn_comm_groups["tp_attn"], dtype)

    def forward(self, bsz, seqlen, ctx_len, stats):
        batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        local_bsz = len(batch_ids)

        for opname in self.ops:
            if isinstance(self.ops[opname], SelfAttention):
                self.ops[opname].forward(local_bsz, seqlen, ctx_len, stats=stats)
            else:
                self.ops[opname].forward(local_bsz*seqlen, stats=stats)
                
    def memory_footprint(self, bsz, ctx_len):
        batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        local_bsz = len(batch_ids)

        mem_size = sum([self.ops[opname].memory_footprint(local_bsz, ctx_len) for opname in self.ops])
        return mem_size # in bytes


class MLANaiveBlock(Layer):
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

        if dist_info.moe_comm == "alltoall":
            self.a2a_dispatch = AlltoAll(uid+"_a2a_disp", hidden_size, dist_info.num_nodes, dist_info, dtype)

    def forward(self, x, ctx_len, stats):
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
            if self.dist_info.moe_comm == "alltoall":
                self.a2a_dispatch.forward(local_bsz*seqlen, stats=stats)
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


class MLABlock(Layer):
    def __init__(self, uid, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype) -> None:
        super().__init__()
        logging.info("Creating MLA block {}".format(uid))

        self.uid = uid
        self.dist_info = dist_info

        self.MLA_naive = MLANaiveBlock(uid+"_naive", hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype)
        self.MLA_absorb = MLAAbsorbBlock(uid+"_absorb", hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype)

    def forward(self, x, ctx_len, stats):
        is_prefill = ctx_len == 0

        if is_prefill:
            return self.MLA_naive.forward(x, ctx_len, stats)
        else:
            return self.MLA_absorb.forward(x, ctx_len, stats)

    def set_next_layer(self, next_layer):
        self.MLA_naive.next_layer = next_layer
        self.MLA_absorb.next_layer = next_layer

    def memory_footprint(self, bsz, ctx_len):
        mem_size = self.MLA_absorb.memory_footprint(bsz, ctx_len)
        return mem_size # in bytes



class LlamaDecodeLayer(Layer):
    def __init__(self, layer_id, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating Decode layer {}".format(layer_id))

        self.dist_info = dist_info

        self.attention = GQABlock(layer_id+"_attn", hidden_size, num_attention_heads, num_key_value_heads, dist_info, dtype)
        self.ffn = FFN(layer_id+"_ffn", hidden_size, intermediate_size, dist_info, dtype)

    def forward(self, queries, ctx_len, stats):
        bsz, seqlen, _ = queries.dims
        self.attention.forward(bsz, seqlen, ctx_len, stats=stats)

        seqlen_per_device_ffn = intceil(seqlen/self.dist_info.sp) # This is only effective in prefill, seqlen=1 in decode anyway
        self.ffn.forward(bsz*seqlen_per_device_ffn, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        bsz_per_device_attn = intceil(bsz/self.dist_info.dp_attn)
        mem_size = self.attention.memory_footprint(bsz_per_device_attn, ctx_len)

        bsz_per_device_ffn = intceil(bsz/self.dist_info.dp_ffn)
        mem_size += self.ffn.memory_footprint(bsz_per_device_ffn)

        return mem_size # in bytes

class DSv3DecodeLayer(Layer):
    def __init__(self, layer_id, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, intermediate_size, moe_intermediate_size, num_experts_per_tok, n_experts, n_shared_experts, dist_info, dtype, is_moe=False) -> None:
        super().__init__()
        logging.info("Creating Decode layer {}".format(layer_id))

        self.layer_id = layer_id
        self.dist_info = dist_info

        self.attention = MLABlock(layer_id+"_attn", hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer=None, dtype=dtype)

        if is_moe:
            self.ffn = MoE(layer_id+"_moe", hidden_size, moe_intermediate_size, num_experts_per_tok, n_experts, n_shared_experts, dist_info, dtype)
        else:
            # self.ffn = Dense(layer_id+"_dense", hidden_size, intermediate_size, dist_info, dtype)
            self.ffn = FFN(layer_id+"_dense", hidden_size, intermediate_size, dist_info, dtype, is_dense_layer=True)

        self.attention.set_next_layer(self.ffn)

    def forward(self, x, ctx_len, stats):
        bsz, seqlen, _ = x.dims

        is_prefill = ctx_len == 0
        if not is_prefill:
            assert seqlen == 1

        # x = self.attention.forward(x, ctx_len, stats=stats)

        self.ffn.forward(x, stats=stats)

        return Tensor(self.layer_id+"_out", self.dist_info.rank, x.dims)

    def memory_footprint(self, bsz, ctx_len):
        batch_ids = self.dist_info.get_local_batchids("attn")
        bsz_per_device_attn = len(batch_ids)
        mem_size = self.attention.memory_footprint(bsz_per_device_attn, ctx_len)

        batch_ids = self.dist_info.get_local_batchids("ffn")
        bsz_per_device_ffn = len(batch_ids)
        mem_size += self.ffn.memory_footprint(bsz_per_device_ffn)

        return mem_size # in bytes

class LMHead(Layer):
    def __init__(self, layer_id, hidden_size, vocab_size, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating LMHead layer {}".format(layer_id))

        vocab_size_per_device = intceil(vocab_size/dist_info.num_nodes)
        self.head = Linear(uid=layer_id+"_head", rank=dist_info.rank, in_features=hidden_size, out_features=vocab_size_per_device, dtype=dtype)

    def forward(self, x, stats):
        self.head.forward(x, stats=stats)

    def memory_footprint(self):
        mem_size = self.head.memory_footprint()
        return mem_size # in bytes