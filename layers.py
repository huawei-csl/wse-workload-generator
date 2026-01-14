import logging
from typing import List

import compute_graph
from compute_graph import get_compute_graph

from tensor import Tensor, get_tensor, View, Split, Concat, Slice, Transpose
from utils import dtype_to_byte, intceil, divide_equal, hash_string
from workload import get_moe_gate_model
import numpy as np

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
    
class Linear(Layer):
    def __init__(self, uid, in_features, out_features, dist_info, dtype) -> None:
        super().__init__()
        logging.debug("Linear layer {} with weight dims: {} x {}".format(uid, in_features, out_features))

        self.uid = uid 
        self.in_features = in_features
        self.out_features = out_features
        self.dist_info = dist_info
        self.dtype = dtype

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        assert hidden_dim == self.in_features, "Input hidden dim {} does not match in_features {}".format(hidden_dim, self.in_features)

        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops(bsz*seqlen)
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims(bsz*seqlen)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        out = Tensor(f"{self.uid}_out", self.dist_info.rank, [bsz, seqlen, self.out_features]) # squeeze seqlen
        stats.append(self.uid, "Linear", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, bsz=None, ctx_len=None):
        memory_footprint =  self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return memory_footprint # weights only, in bytes
    
    def get_dims(self, bsz):
        input_dims = [bsz, self.in_features]
        weight_dims = [self.in_features, self.out_features]
        out_dims = [bsz, self.out_features]
        return str(input_dims) + " x " + str(weight_dims) + " -> " + str(out_dims)
    
    def num_ops(self, bsz):
        n_ops = bsz * self.in_features * self.out_features
        return n_ops # in terms of number of MACs

    def hbm_reads(self):
        rw = self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return rw # weights only, in bytes

    def network_data(self):
        return 0

'''
Equivalent to n_groups Linear layers running in parallel. 
For example: einsum(bshc,hcd->bshd), h is common in all terms, meaning the same computation is repeated for h times. Therefore, h: n_groups
'''
class GroupedLinear(Layer):
    def __init__(self, uid, n_groups, in_features, out_features, dist_info, dtype) -> None:
        super().__init__()
        logging.debug("GroupedLinear layer {} with n_groups: {} weight dims: {} x {}".format(uid, n_groups, in_features, out_features))

        self.uid = uid 
        self.n_groups = n_groups
        self.in_features = in_features
        self.out_features = out_features
        self.dist_info = dist_info
        self.dtype = dtype

    def forward(self, x, stats=None):
        # bsz, seqlen, n_local_heads, hidden_dim = x.dims
    
        # assert n_local_heads == self.n_groups, "Input n_local_heads {} does not match n_groups {}".format(n_local_heads, self.n_groups)
        # assert hidden_dim == self.in_features, "Input hidden dim {} does not match in_features {}".format(hidden_dim, self.in_features)

        assert len(x.dims) == 3, "Input tensor must have 3 dimensions"
        n_groups, batch_dim, in_features = x.dims
    
        assert n_groups == self.n_groups, "Input dim0 {} does not match self.n_groups {}".format(n_groups, self.n_groups)
        assert in_features == self.in_features, "Input in_features {} does not match self.in_features {}".format(in_features, self.in_features)

        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops(batch_dim)
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims(batch_dim)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))

        out = Tensor(f"{self.uid}_out", self.dist_info.rank, [n_groups, batch_dim, self.out_features])
        stats.append(self.uid, "GroupedLinear", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)
        return out

    def memory_footprint(self, bsz=None, ctx_len=None):
        memory_footprint =  self.n_groups * self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return memory_footprint # weights only, in bytes
    
    def num_ops(self, bsz):
        n_ops = self.n_groups * bsz * self.in_features * self.out_features
        return n_ops # in terms of number of MACs

    def hbm_reads(self):
        rw = self.n_groups * self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return rw # weights only, in bytes

    def get_dims(self, bsz):
        input_dims = [self.n_groups, bsz, self.in_features]
        weight_dims = [self.n_groups, self.in_features, self.out_features]
        out_dims = [self.n_groups, bsz, self.out_features]
        return str(input_dims) + " x " + str(weight_dims) + " -> " + str(out_dims)
    
    def network_data(self):
        return 0


class Allreduce(Layer):
    def __init__(self, uid, vector_size, comm_group, dist_info, dtype) -> None:
        super().__init__()
        logging.debug("Allreduce layer {} with vector size: {} ".format(uid, vector_size))

        self.uid = uid
        self.vector_size = vector_size
        self.dtype = dtype
        self.comm_group = comm_group
        self.dist_info = dist_info

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data(bsz*seqlen)
        dims = self.get_dims(bsz*seqlen)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        
        out = Tensor(f"{self.uid}_out", self.dist_info.rank, (bsz, seqlen, hidden_dim))
        stats.append(self.uid, "AllReduce", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.comm_group, dims=dims)
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
        vecsize = 2 * 2 * bsz * self.vector_size * dtype_to_byte(self.dtype) # Reduce: 1 vec receive + 1 vec send, Gather: 1 vec receive + 1 vec send
        logging.debug("{}: network data size (send + receive): {} B".format(self.uid, vecsize))
        return vecsize # in bytes

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
    def __init__(self, uid, vector_size, src: int, dst: int, dtype) -> None:
        super().__init__()
        logging.debug("Unicast layer {} with vector size: {} src:{} dst:{}".format(uid, vector_size, src, dst))

        self.uid = uid
        self.vector_size = vector_size
        self.dtype = dtype
        self.src = src
        self.dst = dst

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        assert x.numel() == self.vector_size, "Input vector size {} does not match expected vector size {}".format(x.numel(), self.vector_size)
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data()
        dims = self.get_dims()

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, network data: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, network_data, dims))
        
        out = Tensor(f"{self.uid}", self.dst, (bsz, seqlen, hidden_dim))
        stats.append(self.uid, "Unicast", memory_footprint, num_ops, hbm_reads, network_data, comm_group=self.dst, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def get_dims(self):
        vec_dims = [self.vector_size,]
        return str(vec_dims)
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self):
        vecsize = self.vector_size * dtype_to_byte(self.dtype) # a vec of this size is sent from a single source to multiple destionations
        logging.debug("{}: network data size: {} B".format(self.uid, vecsize))
        return vecsize # in bytes

class Multicast(Layer):
    def __init__(self, uid, vector_size, src: int, dst: List[int], dtype) -> None:
        super().__init__()
        logging.debug("Multicast layer {} with vector size: {} src:{} dst:{}".format(uid, vector_size, src, dst))

        self.uid = uid
        self.vector_size = vector_size
        self.dtype = dtype
        self.src = src
        self.dst = dst

    def forward(self, x, stats=None):
        bsz, seqlen, hidden_dim = x.dims
        assert x.numel() == self.vector_size, "Input vector size {} does not match expected vector size {}".format(x.numel(), self.vector_size)
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
        vec_dims = [self.vector_size,]
        return str(vec_dims)
    
    def num_ops(self):
        return 0

    def hbm_reads(self):
        return 0
    
    def network_data(self):
        vecsize = self.vector_size * dtype_to_byte(self.dtype) # a vec of this size is sent from a single source to multiple destionations
        logging.debug("{}: network data size: {} B".format(self.uid, vecsize))
        return vecsize # in bytes

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

class MLAAbsorbAttention(Layer):
    def __init__(self, uid, n_local_heads, kv_lora_rank, qk_rope_head_dim, dist_info, dtype) -> None:
        super().__init__()

        self.uid = uid 
        self.n_local_heads = n_local_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        # self.seq_parallel = seq_parallel
        self.dtype = dtype
        self.dist_info = dist_info

    def forward(self, x, ctx_len=None, stats=None):
        bsz, seqlen, n_local_heads, D = x.dims
        assert n_local_heads == self.n_local_heads, "Input n_local_heads {} does not match n_local_heads {}".format(n_local_heads, self.n_local_heads)
        assert D == self.kv_lora_rank + self.qk_rope_head_dim, "Input head dim {} does not match kv_lora_rank+qk_rope_head_dim {}".format(D, self.kv_lora_rank+self.qk_rope_head_dim)

        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)
        dims = self.get_dims(bsz, seqlen, ctx_len)

        logging.debug("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B, dims: {}".format(self.uid, memory_footprint, num_ops, hbm_reads, dims))
        out = Tensor(f"{self.uid}_out", self.dist_info.rank, (bsz, seqlen, n_local_heads, self.kv_lora_rank))
        stats.append(self.uid, "MLAAbsorbAttention", memory_footprint, num_ops, hbm_reads, network_data, comm_group=None, dims=dims)
        get_compute_graph().add_node(self, [x], [out], attrs=None)

        return out

    def memory_footprint(self, bsz, ctx_len):
        ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
        memory_footprint = bsz * ctx_len_per_device * self.kv_lora_rank * dtype_to_byte(self.dtype) # kv_cache
        memory_footprint += bsz * ctx_len_per_device * self.qk_rope_head_dim * dtype_to_byte(self.dtype) # pe_cache
        return memory_footprint  # KV-cache only, in bytes
    
    def get_dims(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.dist_info.sp)
            input_dims = [bsz, seqlen, self.n_local_heads, self.kv_lora_rank]
            K_dims = [bsz, seqlen_per_device, self.kv_lora_rank]
            V_dims = [bsz, seqlen_per_device, self.qk_rope_head_dim]
            out_dims = [bsz, seqlen, self.n_local_heads, self.kv_lora_rank]
        else:
            ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
            input_dims = [bsz, 1, self.n_local_heads, self.kv_lora_rank]
            K_dims = [bsz, ctx_len_per_device, self.kv_lora_rank]
            V_dims = [bsz, ctx_len_per_device, self.qk_rope_head_dim]
            out_dims = [bsz, 1, self.n_local_heads, self.kv_lora_rank]
        return "Q: " + str(input_dims) + ", KV: " + str(K_dims) + ", PE: " + str(V_dims) + " -> " + str(out_dims)
    
    def num_ops(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.dist_info.sp)
            n_ops = bsz * seqlen_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bshc,btc→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.qk_rope_head_dim * seqlen # einsum(bshr,btr→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bsht,btc→bshc)
        else:
            ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
            n_ops = bsz * ctx_len_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bshc,btc→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.qk_rope_head_dim * seqlen # einsum(bshr,btr→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bsht,btc→bshc)
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        ctx_len_per_device = intceil(ctx_len/self.dist_info.sp)
        rw = bsz * ctx_len_per_device * self.kv_lora_rank * dtype_to_byte(self.dtype) # kv_cache
        rw += bsz * ctx_len_per_device * self.qk_rope_head_dim * dtype_to_byte(self.dtype) # pe_cache
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

        self.ops = {}
        self.ops["q_proj"] = Linear(uid+"_qproj", hidden_size, num_heads_per_device * head_dim, dist_info, dtype)
        self.ops["k_proj"] = Linear(uid+"_kproj", hidden_size, num_kv_heads_per_device * head_dim, dist_info, dtype)
        self.ops["v_proj"] = Linear(uid+"_vproj", hidden_size, num_kv_heads_per_device * head_dim, dist_info, dtype)

        self.ops["self_attn"] = SelfAttention(uid+"_selfattn", num_heads_per_device, num_kv_heads_per_device, head_dim, dist_info.sp, dtype)
        if dist_info.sp > 1:
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", num_heads_per_device * head_dim, dist_info.attn_comm_groups["sp"], dist_info, dtype)

        self.ops["o_proj"] = Linear(uid+"_oproj", num_heads_per_device * head_dim, hidden_size, dist_info, dtype)

        if dist_info.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", hidden_size, dist_info.attn_comm_groups["tp_attn"], dist_info, dtype)

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

        n_local_heads = intceil(n_heads / dist_info.tp_attn)

        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.ops = {}
        self.ops["wq_a"] = Linear(uid+"_wqa", hidden_size, q_lora_rank, dist_info, dtype)
        self.ops["wq_b"] = Linear(uid+"_wqb", q_lora_rank, n_local_heads*qk_head_dim, dist_info, dtype)
        self.ops["wkv_a"] = Linear(uid+"_wkva", hidden_size, kv_lora_rank+qk_rope_head_dim, dist_info, dtype)
        self.ops["wkv_b"] = Linear(uid+"_wkvb", kv_lora_rank, n_local_heads*(qk_nope_head_dim + v_head_dim), dist_info, dtype)

        self.ops["naive_attn"] = MLANaiveAttention(uid+"_naiveattn", n_local_heads, qk_head_dim, v_head_dim, dist_info, dtype)
        if dist_info.sp > 1:
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", n_local_heads * v_head_dim, dist_info.attn_comm_groups["sp"], dist_info, dtype)

        self.ops["wo"] = Linear(uid+"_wo", n_local_heads*v_head_dim, hidden_size, dist_info, dtype)
        if dist_info.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", hidden_size, dist_info.attn_comm_groups["tp_attn"], dist_info, dtype)

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


class MLAAbsorbBlock(Layer):
    def __init__(self, uid, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dist_info, next_layer, dtype) -> None:
        super().__init__()
        logging.info("Creating MLA naive layer {}".format(uid))

        self.uid = uid
        self.dist_info = dist_info
        self.next_layer = next_layer
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.n_local_heads = intceil(n_heads / dist_info.tp_attn)
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim

        self.ops = {}
        self.ops["wq_a"] = Linear(uid+"_wqa", hidden_size, q_lora_rank, dist_info, dtype)
        self.ops["wq_b"] = Linear(uid+"_wqb", q_lora_rank, self.n_local_heads*self.qk_head_dim, dist_info, dtype)
        self.ops["wkv_a"] = Linear(uid+"_wkva", hidden_size, kv_lora_rank+qk_rope_head_dim, dist_info, dtype)
        self.ops["wkv_b1"] = GroupedLinear(uid+"_wkvb1", self.n_local_heads, qk_nope_head_dim, kv_lora_rank, dist_info, dtype)
        self.ops["absorb_attn"] = MLAAbsorbAttention(uid+"_absorbattn", self.n_local_heads, kv_lora_rank, qk_rope_head_dim, dist_info, dtype)
        if dist_info.sp > 1:
            # Allreduce to aggregate attention output from sequence parallel devices
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", self.n_local_heads * qk_rope_head_dim, dist_info.attn_comm_groups["sp"], dist_info, dtype)
        
        self.ops["wkv_b2"] = GroupedLinear(uid+"_wkvb2", self.n_local_heads, kv_lora_rank, v_head_dim, dist_info, dtype)
        self.ops["wo"] = Linear(uid+"_wo", self.n_local_heads*v_head_dim, hidden_size, dist_info, dtype)
        if dist_info.tp_attn > 1:
            # Allreduce to aggregate output from tensor parallel devices
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", hidden_size, dist_info.attn_comm_groups["tp_attn"], dist_info, dtype)

    def forward(self, x, ctx_len, stats):
        local_bsz, seqlen, _ = x.dims
        assert seqlen == 1, "Absorb block should be used only for decode"

        kv = self.ops["wkv_a"].forward(x, stats=stats)
        #TODO: implement KV update

        q = self.ops["wq_a"].forward(x, stats=stats)
        q = self.ops["wq_b"].forward(q, stats=stats)

        q = View(q, [local_bsz, seqlen, self.n_local_heads, self.qk_head_dim]).forward(stats=stats)

        q_nope, q_pe = Split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1).forward(stats=stats)
        
        q_nope = View(q_nope, [local_bsz*seqlen, self.n_local_heads, self.qk_nope_head_dim]).forward(stats=stats)

        q_nope = Transpose(q_nope, [0, 1]).forward(stats=stats)

        q_nope = self.ops["wkv_b1"].forward(q_nope, stats=stats)

        q_nope = Transpose(q_nope, [0, 1]).forward(stats=stats)

        q_nope = View(q_nope, [local_bsz, seqlen, self.n_local_heads, self.kv_lora_rank]).forward(stats=stats)

        q = Concat([q_nope, q_pe], axis=-1).forward(stats=stats)

        attn_out = self.ops["absorb_attn"].forward(q, ctx_len, stats=stats)
        attn_out = Transpose(attn_out, [0, 2]).forward(stats=stats)

        if self.dist_info.sp > 1:
            attn_out = View(attn_out, [local_bsz, seqlen, self.n_local_heads*self.kv_lora_rank]).forward(stats=stats)
            attn_out = self.ops["allreduce_sp"].forward(attn_out, stats=stats)
            attn_out = View(attn_out, [local_bsz, seqlen, self.n_local_heads, self.kv_lora_rank]).forward(stats=stats)

        attn_out = View(attn_out, [self.n_local_heads, local_bsz*seqlen, self.kv_lora_rank]).forward(stats=stats)
        x = self.ops["wkv_b2"].forward(attn_out, stats=stats)
        x = Transpose(x, [0, 1]).forward(stats=stats)

        x = View(x, [local_bsz, seqlen, self.n_local_heads, self.v_head_dim]).forward(stats=stats)
        x = View(x, [local_bsz, seqlen, self.n_local_heads*self.v_head_dim]).forward(stats=stats)
        y = self.ops["wo"].forward(x, stats=stats)

        if self.dist_info.tp_attn > 1:
            y = self.ops["allreduce_tp"].forward(y, stats=stats)

        return y

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
            raise NotImplementedError("Next layer type {} not implemented for MLAAbsorbBlock".format(type(self.next_layer)))

    def memory_footprint(self, bsz, ctx_len):
        # batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        batch_ids = self.dist_info.get_local_batchids("attn")
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

# Each expert is an instance of this layer
class FFN(Layer):
    def __init__(self, uid, hidden_size, intermediate_size, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating FFN layer {}".format(uid))
        self.uid = uid

        self.dist_info = dist_info
        inter_size_per_node = intceil(intermediate_size/dist_info.tp_ffn) 

        self.ops = {}
        self.ops["up"] = Linear(uid+"_up", hidden_size, inter_size_per_node, dist_info, dtype)
        self.ops["gate"] = Linear(uid+"_gate", hidden_size, inter_size_per_node, dist_info, dtype)
        self.ops["down"] = Linear(uid+"_down", inter_size_per_node, hidden_size, dist_info, dtype)
        
        # in case we use TP for experts
        if dist_info.tp_ffn > 1:
            self.ops["allreduce"] = Allreduce(uid+"_ar", hidden_size, dist_info.ffn_comm_groups["tp_ffn"], dist_info, dtype)

    def forward(self, x, stats=None):
        x1 = self.ops["up"].forward(x, stats=stats)
        x2 = self.ops["gate"].forward(x, stats=stats)
        #TODO: implement element-wise multiplication and activation

        y = self.ops["down"].forward(x1, stats=stats)

        if self.dist_info.tp_ffn > 1:
            y = self.ops["allreduce"].forward(y, stats=stats)
        
        return y


    def memory_footprint(self, bsz=None, ctx_len=None):
        mem_size = sum([self.ops[opname].memory_footprint() for opname in self.ops])
        return mem_size # in bytes


'''
Same as FFN but assumes TP=EP. This is used in first 3 layers in DSv3
'''
class Dense(Layer):
    def __init__(self, uid, hidden_size, intermediate_size, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating Dense layer {}".format(uid))
        self.uid = uid

        self.dist_info = dist_info
        inter_size_per_node = intceil(intermediate_size/dist_info.num_nodes) 

        self.ops = {}
        self.ops["up"] = Linear(uid+"_up", hidden_size, inter_size_per_node, dist_info, dtype)
        self.ops["gate"] = Linear(uid+"_gate", hidden_size, inter_size_per_node, dist_info, dtype)
        self.ops["down"] = Linear(uid+"_down", inter_size_per_node, hidden_size, dist_info, dtype)
        
        if dist_info.num_nodes > 1: # we assume tp=ep for these layers
            self.ops["allreduce"] = Allreduce(uid+"_ar", hidden_size, dist_info.ffn_comm_groups["ep"], dist_info, dtype)

    def forward(self, x, stats=None):
        x1 = self.ops["up"].forward(x, stats=stats)
        x2 = self.ops["gate"].forward(x, stats=stats)
        #TODO: implement element-wise multiplication and activation

        y = self.ops["down"].forward(x1, stats=stats)

        if self.dist_info.num_nodes > 1:
            y = self.ops["allreduce"].forward(y, stats=stats)
        
        return y

    def memory_footprint(self, bsz, ctx_len=None):
        mem_size = sum([self.ops[opname].memory_footprint(bsz) for opname in self.ops])
        return mem_size # in bytes
    
class MoE(Layer):
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

        if self.dist_info.ep > 1 and dist_info.moe_comm == "alltoall":
            self.a2a_dispatch = AlltoAll(uid+"_a2a_disp", hidden_size, dist_info.num_nodes, dist_info, dtype)

        if self.dist_info.ep > 1 and self.dist_info.moe_comm == "alltoall":
            self.a2a_combine = AlltoAll(uid+"_a2a_comb", hidden_size, self.dist_info.num_nodes, dist_info, dtype)

        self.expertid_to_node = self.dist_info.get_expert_mapping(self.n_experts)
        local_experts = [expert_id for expert_id in range(self.n_experts) if self.expertid_to_node[expert_id] == self.dist_info.rank_ep]

        self.experts = {}
        for i in local_experts:
            self.experts[i] = FFN(uid+"_exp_"+str(i), hidden_size, moe_intermediate_size, dist_info, dtype)
        
        intermediate_size = moe_intermediate_size * n_shared_experts

        self.shared_expert = None
        if self.dist_info.rank in self.dist_info.shared_expert_ranks:
            self.shared_expert = FFN(uid+"_shared_exp", hidden_size, intermediate_size, dist_info, dtype)


    def forward(self, x, stats):
        bsz, seqlen, hidden_dim = x.dims

        if self.dist_info.moe_comm == "alltoall":
            x_all = Tensor(x.uid + f"_dispatch", self.dist_info.rank, [get_moe_gate_model().global_bsz, seqlen, hidden_dim])
            self.a2a_dispatch.forward(x_all, stats=stats)
        elif self.dist_info.moe_comm == "multicast":
            
            # after attention allreduce, each node within the DP cluster has identical data
            # therefore, only the master node in the DP cluster dispatches the data to experts
            # TODO: distribute this task to all nodes in the DP cluster for better load balancing
            if self.dist_info.is_dp_master():
                
                # batch ids processed by this DP cluster
                batch_ids = self.dist_info.get_local_batchids("attn")

                for batch_id in batch_ids:
                    # get expert ids for this query
                    mapping = get_moe_gate_model().get_mapping_by_batchids(self.uid, batch_id)
                    logging.debug("batch_id: {}, mapping: {}".format(batch_id, mapping))

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

                    x_slice = Slice(x, [batch_id], axis=0).forward(stats=stats)
                    if len(dst_nodes) > 0:
                        Multicast(self.uid+"_multicast_exp_"+str(batch_id), vector_size=self.hidden_size, src=self.dist_info.rank, dst=dst_nodes, dtype=self.dtype).forward(
                            # x.slice([batch_id], axis=0), stats=stats)
                            x_slice, stats=stats)
                    
                    # x_local = NoOp(x_slice, uid=x_slice.uid + f"_dst{self.dist_info.rank}").forward(stats=stats)

                Barrier(self.uid+"_barrier", nodes=list(range(self.dist_info.num_nodes))).forward(stats=stats) # ensure all nodes have received the multicast before proceeding
        else:
            raise NotImplementedError("MoE communication method {} not implemented".format(self.dist_info.moe_comm))

        recv_batch_ids = {}
        exp_outs = {}
        total_moe_num_tokens_per_device = 0
        for e in self.experts:
            expert_routings = get_moe_gate_model().get_expert_routings(layer_id=self.uid)
            recv_batch_ids[e] = sorted(np.where(expert_routings==e)[1])

            logging.debug("expert {} num of routed samples: {}".format(e, len(recv_batch_ids[e])))

            if len(recv_batch_ids[e]) > 0:
                # Give a unique uid to the concat operation, based on indices
                concat_uid = self.uid + "_dispatch_recv_concat_" + hash_string("_".join([str(batch_id) for batch_id in recv_batch_ids[e]]))
                x_recv = Concat(
                    [Tensor(f"{x.uid}_slice{batch_id}", self.dist_info.rank, [1, seqlen, hidden_dim]) for batch_id in recv_batch_ids[e]], 
                    axis=0,
                    uid=concat_uid).forward(stats=stats)
                exp_outs[e] = self.experts[e].forward(x_recv, stats=stats)

            total_moe_num_tokens_per_device += len(recv_batch_ids[e])

        logging.debug("Total number of routed samples for device {}: {}".format(self.dist_info.rank_ep, total_moe_num_tokens_per_device))

        # if this node holds a copy of the shared expert
        if self.shared_expert:
            batch_ids_for_shared = [batch_id for batch_id, mapped_shared in self.dist_info.batch_to_shared_exp.items() if self.dist_info.rank == mapped_shared]
            # x_for_shared = concat([Tensor(x.uid + f"_slice{batch_id}", [1, seqlen, hidden_dim]) for batch_id in batch_ids_for_shared], axis=0)
            concat_uid = self.uid + "_shared_concat_" + hash_string("_".join([str(batch_id) for batch_id in batch_ids_for_shared]))
            x_for_shared = Concat(
                [Tensor(x.uid + f"_slice{batch_id}", self.dist_info.rank, [1, seqlen, hidden_dim]) for batch_id in batch_ids_for_shared], 
                axis=0, 
                uid=concat_uid).forward(stats=stats)

            self.shared_expert.forward(x_for_shared, stats=stats)
            logging.debug("Shared expert on node {} processed {} samples".format(self.dist_info.rank, len(batch_ids_for_shared)))

        if self.dist_info.ep > 1:
            if self.dist_info.moe_comm == "alltoall":
                x_all = Tensor(x.uid + f"_combine", self.dist_info.rank, [get_moe_gate_model().global_bsz*(self.num_experts_per_tok+self.n_shared_experts), seqlen, hidden_dim])
                out_tensor = self.a2a_combine.forward(x_all, stats=stats)
            elif self.dist_info.moe_comm == "multicast":
                # after MoE layer, gather the outputs in specific nodes to sum them
                # at the moment, we gather them at the dp master of each DP cluster
                # we merge all unicasts to the same dst node
                # TODO: distribute this task to all nodes in the DP cluster for better load balancing

                batchid_dst = {i: [] for i in range(self.dist_info.num_nodes)}

                for e in self.experts:
                    mapping = get_moe_gate_model().get_expert_routings(self.uid)

                    # batch ids routed to expert e
                    batch_ids = np.nonzero(mapping==e)[1]

                    # which dp cluster the dst node belongs to
                    dp_ranks = self.dist_info.get_dp_rank_from_batchids(batch_ids, "attn")

                    # we send the expert outputs to the dp master node of the corresponding dp cluster
                    dst_nodes = [self.dist_info.get_dp_master(dp_rank, "attn") for dp_rank in dp_ranks]
                    
                    for batch_id, dst_node in zip(batch_ids, dst_nodes):
                        batchid_dst[dst_node].append((batch_id, e))

                logging.debug("gather expert outputs at dst_nodes: {}".format(batchid_dst))

                # sort the batchids for each dst_node, first by the expert_id then by batch_id
                for dst_node in batchid_dst:
                    batchid_dst[dst_node] = sorted(batchid_dst[dst_node], key=lambda x: (x[1], x[0]) ) # sort by batch_id

                for dst_node in batchid_dst:
                    # if dst node is itself, skip
                    if dst_node == self.dist_info.rank:
                        continue
                    
                    # if there are tokens to send to node i, do unicast
                    if len(batchid_dst[dst_node]) > 0:
                        # unicast_tensor = concat([x.slice([batch_id], axis=0, uid=x.uid + f"_slice{batch_id}") for batch_id in batchid_dst[i]], axis=0)
                        concat_uid = self.uid + "_gather_unicast_concat_" + hash_string("_".join([f"{batch_id}e{expert_id}" for batch_id, expert_id in batchid_dst[dst_node]]))
                        unicast_tensor = Concat(
                            # [x.slice([batch_id], axis=0, uid=x.uid + f"_slice{batch_id}") for batch_id in batchid_dst[i]], 
                            [Slice(exp_outs[expert_id], [batch_id], axis=0).forward(stats=stats) for batch_id, expert_id in batchid_dst[dst_node]], 
                            axis=0,
                            uid=concat_uid).forward(stats=stats)
                        Unicast(self.uid+"_unicast_"+str(dst_node), vector_size=unicast_tensor.numel(), src=self.dist_info.rank, dst=dst_node, dtype=self.dtype).forward(
                            unicast_tensor, stats=stats)

                Barrier(self.uid+"_barrier_uc", nodes=list(range(self.dist_info.num_nodes))).forward(stats=stats) # ensure all nodes have received the unicast before proceeding

                batch_ids = self.dist_info.get_local_batchids("attn")
                out_tensor = Tensor(self.uid + "_out", self.dist_info.rank, [len(batch_ids), seqlen, hidden_dim])
                # out_tensor = concat([x.slice([batch_id], axis=0, uid=x.uid + f"_slice[{batch_id}]") for batch_id in batch_ids], axis=0)

                # once all unicasts are done, perform a multicast within the DP cluster for the next layer
                # batch_ids = get_itemids_from_bucketid(self.dist_info.rank, bsz, self.dist_info.num_nodes)
                if self.dist_info.is_dp_master():
                    Multicast(self.uid+"_multicast_dp", vector_size=self.hidden_size*len(batch_ids), src=self.dist_info.rank, dst=self.dist_info.dp_attn_cluster, dtype=self.dtype).forward(
                        out_tensor, stats=stats)
                
                Barrier(self.uid+"_barrier_mc", nodes=self.dist_info.dp_attn_cluster).forward(stats=stats) # ensure all nodes in the DP cluster have received the multicast before proceeding
            else:
                raise NotImplementedError("MoE communication method {} not implemented".format(self.dist_info.moe_comm))

        return out_tensor

    def memory_footprint(self, bsz=None, ctx_len=None):
        mem_size = sum([self.experts[e].memory_footprint() for e in self.experts])
        if self.shared_expert:
            mem_size += self.shared_expert.memory_footprint()
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
        # batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        # bsz_per_device_attn = len(batch_ids)
        # bsz_per_device_attn = intceil(bsz/self.dist_info.dp_attn)
        # self.attention.forward(bsz_per_device_attn, seqlen, ctx_len, stats=stats)
        self.attention.forward(bsz, seqlen, ctx_len, stats=stats)

        # bsz_per_device_ffn = intceil(bsz/self.dist_info.dp_ffn)
        seqlen_per_device_ffn = intceil(seqlen/self.dist_info.sp) # This is only effective in prefill, seqlen=1 in decode anyway
        # self.ffn.forward(bsz_per_device_ffn*seqlen_per_device_ffn, stats=stats)
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
            self.ffn = Dense(layer_id+"_dense", hidden_size, intermediate_size, dist_info, dtype)

        self.attention.set_next_layer(self.ffn)

    def forward(self, x, ctx_len, stats):
        bsz, seqlen, _ = x.dims

        is_prefill = ctx_len == 0
        if not is_prefill:
            assert seqlen == 1

        x = self.attention.forward(x, ctx_len, stats=stats)

        self.ffn.forward(x, stats=stats)

        return Tensor(self.layer_id+"_out", self.dist_info.rank, x.dims)

    def memory_footprint(self, bsz, ctx_len):
        # batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_attn, bsz, self.dist_info.dp_attn)
        batch_ids = self.dist_info.get_local_batchids("attn")
        bsz_per_device_attn = len(batch_ids)
        # bsz_per_device_attn = intceil(bsz/self.dist_info.dp_attn)
        mem_size = self.attention.memory_footprint(bsz_per_device_attn, ctx_len)

        # batch_ids = get_itemids_from_bucketid(self.dist_info.rank_dp_ffn, bsz, self.dist_info.dp_ffn)
        batch_ids = self.dist_info.get_local_batchids("ffn")
        bsz_per_device_ffn = len(batch_ids)
        # bsz_per_device_ffn = intceil(bsz/self.dist_info.dp_ffn)
        mem_size += self.ffn.memory_footprint(bsz_per_device_ffn)

        return mem_size # in bytes

class LMHead(Layer):
    def __init__(self, layer_id, hidden_size, vocab_size, dist_info, dtype) -> None:
        super().__init__()
        logging.info("Creating LMHead layer {}".format(layer_id))

        vocab_size_per_device = intceil(vocab_size/dist_info.num_nodes)
        self.head = Linear(uid=layer_id+"_head", in_features=hidden_size, out_features=vocab_size_per_device, dist_info=dist_info, dtype=dtype)

    def forward(self, x, stats):
        self.head.forward(x, stats=stats)

    def memory_footprint(self):
        mem_size = self.head.memory_footprint()
        return mem_size # in bytes