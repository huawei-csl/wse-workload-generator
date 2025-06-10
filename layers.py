import logging

from utils import dtype_to_byte, intceil
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
    
class Linear(Layer):
    def __init__(self, uid, in_features, out_features, dtype) -> None:
        super().__init__()
        logging.info("Linear layer {} with weight dims: {} x {}".format(uid, in_features, out_features))

        self.uid = uid 
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

    def forward(self, bsz, seqlen=None, ctx_len=None, stats=None):
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops(bsz, seqlen)
        hbm_reads = self.hbm_reads(bsz)
        network_data = self.network_data(bsz)

        logging.info("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B".format(self.uid, memory_footprint, num_ops, hbm_reads))
        stats.append(self.uid, memory_footprint, num_ops, hbm_reads, network_data)

    def memory_footprint(self, bsz=None, ctx_len=None):
        memory_footprint =  self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return memory_footprint # weights only, in bytes
    
    def num_ops(self, bsz, seqlen, ctx_len=None):
        n_ops = bsz * seqlen * self.in_features * self.out_features
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        rw = self.in_features * self.out_features * dtype_to_byte(self.dtype)
        return rw # weights only, in bytes

    def network_data(self, bsz=None):
        return 0
    
class SelfAttention(Layer):
    def __init__(self, uid, num_attention_heads, num_key_value_heads, head_dim, seq_parallel, dtype) -> None:
        super().__init__()
        logging.info("SelfAttention layer {} with KV-cache dims: bsz x ctx_len x {} x {}".format(uid, num_key_value_heads, head_dim))

        self.uid = uid
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.seq_parallel = seq_parallel

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)

        logging.info("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B".format(self.uid, memory_footprint, num_ops, hbm_reads))
        stats.append(self.uid, memory_footprint, num_ops, hbm_reads, network_data)

    def memory_footprint(self, bsz, ctx_len):
        memory_footprint = 2 * bsz * intceil(ctx_len/self.seq_parallel) * self.num_key_value_heads * self.head_dim * dtype_to_byte(self.dtype) # KV-cache
        return memory_footprint  # KV-cache only, in bytes

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

class Allreduce(Layer):
    def __init__(self, uid, vector_size, dtype) -> None:
        super().__init__()
        logging.info("Allreduce layer {} with vector size: {} ".format(uid, vector_size))

        self.uid = uid
        self.vector_size = vector_size
        self.dtype = dtype

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data(bsz, seqlen)

        stats.append(self.uid, memory_footprint, num_ops, hbm_reads, network_data)

    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def num_ops(self, bsz=None, ctx_len=None):
        return 0

    def hbm_reads(self, bsz=None, ctx_len=None):
        return 0
    
    def network_data(self, bsz, seqlen):
        vecsize = 2 * bsz * seqlen * self.vector_size * dtype_to_byte(self.dtype) # 1 vec receive + 1 vec send
        logging.info("{}: network data size (send + receive): {} B".format(self.uid, vecsize))
        return vecsize # in bytes

class AlltoAll(Layer):
    def __init__(self, uid, vector_size, cluster_size, dtype) -> None:
        super().__init__()
        logging.info("AlltoAll layer {} with vector size: {} among {} devices".format(uid, vector_size, cluster_size))

        self.uid = uid
        self.vector_size = vector_size
        self.cluster_size = cluster_size
        self.dtype = dtype

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        memory_footprint = self.memory_footprint()
        num_ops = self.num_ops()
        hbm_reads = self.hbm_reads()
        network_data = self.network_data(bsz, seqlen)

        stats.append(self.uid, memory_footprint, num_ops, hbm_reads, network_data)

    def memory_footprint(self, bsz=None, ctx_len=None):
        return 0

    def num_ops(self, bsz=None, ctx_len=None):
        return 0

    def hbm_reads(self, bsz=None, ctx_len=None):
        return 0
    
    def network_data(self, bsz, seqlen):
        vecsize = 2 * bsz * seqlen * self.vector_size * self.cluster_size * dtype_to_byte(self.dtype) # N vec receive + N vec send, N: no. of devices in a cluster
        logging.info("{}: network data size (send + receive): {} B".format(self.uid, vecsize))
        return vecsize # in bytes
    
class GQABlock(Layer):
    def __init__(self, uid, hidden_size, num_attention_heads, num_key_value_heads, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating GQA block {}".format(uid))

        assert hidden_size % num_attention_heads == 0
        head_dim = hidden_size // num_attention_heads

        self.system_config = system_config

        num_heads_per_device = intceil(num_attention_heads / system_config.tp_attn)
        num_kv_heads_per_device = intceil(num_key_value_heads / system_config.tp_attn)

        self.uid = uid

        self.ops = {}
        self.ops["q_proj"] = Linear(uid+"_qproj", hidden_size, num_heads_per_device * head_dim, dtype)
        self.ops["k_proj"] = Linear(uid+"_kproj", hidden_size, num_kv_heads_per_device * head_dim, dtype)
        self.ops["v_proj"] = Linear(uid+"_vproj", hidden_size, num_kv_heads_per_device * head_dim, dtype)

        self.ops["self_attn"] = SelfAttention(uid+"_selfattn", num_heads_per_device, num_kv_heads_per_device, head_dim, system_config.sp, dtype)
        if system_config.sp > 1:
            self.ops["allreduce_sp"] = Allreduce(uid+"_ar_sp", num_heads_per_device*head_dim, dtype)

        self.ops["o_proj"] = Linear(uid+"_oproj", num_heads_per_device * head_dim, hidden_size, dtype)

        if system_config.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", hidden_size, dtype)

    def forward(self, bsz, seqlen, ctx_len, stats):
        for opname in self.ops:
            self.ops[opname].forward(bsz, seqlen, ctx_len, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        mem_size = sum([self.ops[opname].memory_footprint(bsz, ctx_len) for opname in self.ops])
        return mem_size # in bytes


class MLANaiveAttention(Layer):
    def __init__(self, uid, n_local_heads, qk_head_dim, v_head_dim, seq_parallel, dtype) -> None:
        super().__init__()

        self.uid = uid 
        self.n_local_heads = n_local_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.seq_parallel = seq_parallel
        self.dtype = dtype

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)

        logging.info("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B".format(self.uid, memory_footprint, num_ops, hbm_reads))
        stats.append(self.uid, memory_footprint, num_ops, hbm_reads, network_data)

    def memory_footprint(self, bsz, ctx_len):
        memory_footprint = bsz * intceil(ctx_len/self.seq_parallel) * self.n_local_heads * self.qk_head_dim * dtype_to_byte(self.dtype) # k_cache
        memory_footprint += bsz * intceil(ctx_len/self.seq_parallel) * self.n_local_heads * self.v_head_dim * dtype_to_byte(self.dtype) # v_cache
        return memory_footprint  # KV-cache only, in bytes

    def num_ops(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.seq_parallel)
            n_ops = bsz * seqlen_per_device * self.n_local_heads * self.qk_head_dim * seqlen # einsum(bshd,bthd→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.v_head_dim * seqlen # einsum(bsht,bthv→bshv)
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            n_ops = bsz * ctx_len_per_device * self.n_local_heads * self.qk_head_dim * seqlen # einsum(bshd,bthd→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.v_head_dim * seqlen # einsum(bsht,bthv→bshv)
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
        rw = bsz * ctx_len_per_device * self.n_local_heads * self.qk_head_dim * dtype_to_byte(self.dtype) # k_cache
        rw += bsz * ctx_len_per_device * self.n_local_heads * self.v_head_dim * dtype_to_byte(self.dtype) # v_cache
        return rw # KV-cache only, in bytes

    def network_data(self, bsz=None):
        return 0

class MLAAbsorbAttention(Layer):
    def __init__(self, uid, n_local_heads, kv_lora_rank, qk_rope_head_dim, seq_parallel, dtype) -> None:
        super().__init__()

        self.uid = uid 
        self.n_local_heads = n_local_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.seq_parallel = seq_parallel
        self.dtype = dtype

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        memory_footprint = self.memory_footprint(bsz, ctx_len)
        num_ops = self.num_ops(bsz, seqlen, ctx_len)
        hbm_reads = self.hbm_reads(bsz, ctx_len)
        network_data = self.network_data(bsz)

        logging.info("{} memory footprint: {} B, n_ops: {} MACs, HBM read: {} B".format(self.uid, memory_footprint, num_ops, hbm_reads))
        stats.append(self.uid, memory_footprint, num_ops, hbm_reads, network_data)

    def memory_footprint(self, bsz, ctx_len):
        memory_footprint = bsz * intceil(ctx_len/self.seq_parallel) * self.kv_lora_rank * dtype_to_byte(self.dtype) # kv_cache
        memory_footprint += bsz * intceil(ctx_len/self.seq_parallel) * self.qk_rope_head_dim * dtype_to_byte(self.dtype) # pe_cache
        return memory_footprint  # KV-cache only, in bytes

    def num_ops(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len == 0
        if is_prefill:
            seqlen_per_device = intceil(seqlen/self.seq_parallel)
            n_ops = bsz * seqlen_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bshc,btc→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.qk_rope_head_dim * seqlen # einsum(bshr,btr→bsht)
            n_ops += bsz * seqlen_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bsht,btc→bshc)
        else:
            ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
            n_ops = bsz * ctx_len_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bshc,btc→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.qk_rope_head_dim * seqlen # einsum(bshr,btr→bsht)
            n_ops += bsz * ctx_len_per_device * self.n_local_heads * self.kv_lora_rank * seqlen # einsum(bsht,btc→bshc)
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        ctx_len_per_device = intceil(ctx_len/self.seq_parallel)
        rw = bsz * ctx_len_per_device * self.kv_lora_rank * dtype_to_byte(self.dtype) # kv_cache
        rw += bsz * ctx_len_per_device * self.qk_rope_head_dim * dtype_to_byte(self.dtype) # pe_cache
        return rw # KV-cache only, in bytes

    def network_data(self, bsz=None):
        return 0
    

class MLANaiveBlock(Layer):
    def __init__(self, uid, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating MLA naive layer {}".format(uid))

        n_local_heads = intceil(n_heads / system_config.tp_attn)

        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.ops = {}
        self.ops["wq_a"] = Linear(uid+"_wqa", hidden_size, q_lora_rank, dtype)
        self.ops["wq_b"] = Linear(uid+"_wqb", q_lora_rank, n_local_heads*qk_head_dim, dtype)
        self.ops["wkv_a"] = Linear(uid+"_wkva", hidden_size, kv_lora_rank+qk_rope_head_dim, dtype)
        self.ops["wkv_b"] = Linear(uid+"_wkvb", kv_lora_rank, n_local_heads*(qk_nope_head_dim + v_head_dim), dtype)
        self.ops["naive_attn"] = MLANaiveAttention(uid+"_naiveattn", n_local_heads, qk_head_dim, v_head_dim, system_config.sp, dtype)
        self.ops["wo"] = Linear(uid+"_wo", n_local_heads*v_head_dim, hidden_size, dtype)
        if system_config.tp_attn > 1:
            self.ops["allreduce_tp"] = Allreduce(uid+"_ar_tp", hidden_size, dtype)
        
    def forward(self, bsz, seqlen, ctx_len, stats):
        for opname in self.ops:
            self.ops[opname].forward(bsz, seqlen, ctx_len, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        mem_size = sum([self.ops[opname].memory_footprint(bsz, ctx_len) for opname in self.ops])
        return mem_size # in bytes



class MLABlock(Layer):
    def __init__(self, uid, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating MLA block {}".format(uid))

        self.uid = uid
        self.system_config = system_config

        self.MLA_naive = MLANaiveBlock(uid+"_naive", hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, system_config, dtype)

    def forward(self, bsz, seqlen, ctx_len, stats):
        self.MLA_naive.forward(bsz, seqlen, ctx_len, stats)

    def memory_footprint(self, bsz, ctx_len):
        mem_size = self.MLA_naive.memory_footprint(bsz, ctx_len)
        return mem_size # in bytes
     
class FFN(Layer):
    def __init__(self, uid, hidden_size, intermediate_size, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating FFN layer {}".format(uid))
        self.uid = uid

        self.system_config = system_config
        assert intermediate_size % system_config.tp_ffn == 0
        inter_size_per_node = intceil(intermediate_size/system_config.tp_ffn) 

        self.ops = {}
        self.ops["up"] = Linear(uid+"_up", hidden_size, inter_size_per_node, dtype)
        self.ops["gate"] = Linear(uid+"_gate", hidden_size, inter_size_per_node, dtype)
        self.ops["down"] = Linear(uid+"_down", inter_size_per_node, hidden_size, dtype)
        
        if system_config.tp_ffn > 1:
            self.ops["allreduce"] = Allreduce(uid+"_ar", hidden_size, dtype)

    def forward(self, bsz, seqlen, ctx_len=None, stats=None):
        for opname in self.ops:
            self.ops[opname].forward(bsz, seqlen, ctx_len, stats=stats)

    def memory_footprint(self, bsz, ctx_len=None):
        mem_size = sum([self.ops[opname].memory_footprint(bsz) for opname in self.ops])
        return mem_size # in bytes

    def num_ops(self, bsz=None):
        n_ops = sum([self.ops[opname].num_ops(bsz) for opname in self.ops])
        return n_ops # in terms of number of MACs
    
    def hbm_reads(self):
        rw = sum([self.ops[opname].hbm_reads() for opname in self.ops])
        return rw # in bytes

    def network_data(self, bsz):
        vecsize = sum([self.ops[opname].network_data(bsz) for opname in self.ops])
        return vecsize # in bytes


class MoE(Layer):
    def __init__(self, uid, hidden_size, moe_intermediate_size, num_experts_per_tok, n_experts, n_shared_experts, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating MoE layer {}".format(uid))
        self.uid = uid

        self.num_experts_per_tok = num_experts_per_tok
        self.n_experts = n_experts
        self.n_shared_experts = n_shared_experts
        self.system_config = system_config

        if self.system_config.ep > 1:
            self.a2a_dispatch = AlltoAll(uid+"_a2a_disp", hidden_size, self.system_config.num_nodes, dtype)
            self.a2a_combine = AlltoAll(uid+"_a2a_comb", hidden_size, self.system_config.num_nodes, dtype)

        rank_ep = system_config.rank_ep
        assert n_experts % system_config.ep == 0
        n_experts_per_device = n_experts // system_config.ep

        self.experts = {}
        for i in range(rank_ep*n_experts_per_device, (rank_ep+1)*n_experts_per_device):
            self.experts[i] = FFN(uid+"_exp_"+str(i), hidden_size, moe_intermediate_size, system_config, dtype)
        
        intermediate_size = moe_intermediate_size * n_shared_experts
        self.shared_expert = FFN(uid+"_shared_exp", hidden_size, intermediate_size, system_config, dtype)

    ## TODO: Do we really need a global all-to-all communication for both dispatch and combine?
    def forward(self, bsz, seqlen, ctx_len, stats):
        self.a2a_dispatch.forward(bsz, seqlen=seqlen, stats=stats)

        for e in self.experts:
            bsz_for_expert_i = get_moe_gate_model().get_bincounts(layer_id=self.uid, expert_id=e)
            logging.info("expert {} num of routed samples: {}".format(e, bsz_for_expert_i))
            if bsz_for_expert_i > 0:
                self.experts[e].forward(bsz_for_expert_i, seqlen=1, stats=stats)
        self.shared_expert.forward(bsz*seqlen, seqlen=1, stats=stats)
        
        self.a2a_combine.forward(bsz, seqlen=seqlen, stats=stats)

class LlamaDecodeLayer(Layer):
    def __init__(self, layer_id, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating Decode layer {}".format(layer_id))

        self.attention = GQABlock(layer_id+"_attn", hidden_size, num_attention_heads, num_key_value_heads, system_config, dtype)
        self.ffn = FFN(layer_id+"_ffn", hidden_size, intermediate_size, system_config, dtype)

    def forward(self, bsz, seqlen, ctx_len, stats):
        bsz_per_device_attn = intceil(bsz/self.system_config.dp_attn)
        self.attention.forward(bsz_per_device_attn, seqlen, ctx_len, stats=stats)

        bsz_per_device_ffn = intceil(bsz/self.system_config.dp_ffn)
        self.ffn.forward(bsz_per_device_ffn, seqlen, ctx_len, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        bsz_per_device_attn = intceil(bsz/self.system_config.dp_attn)
        mem_size = self.attention.memory_footprint(bsz_per_device_attn, ctx_len)

        bsz_per_device_ffn = intceil(bsz/self.system_config.dp_ffn)
        mem_size += self.ffn.memory_footprint(bsz_per_device_ffn)

        return mem_size # in bytes


class DSv3DecodeLayer(Layer):
    def __init__(self, layer_id, hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, intermediate_size, num_experts_per_tok, n_experts, n_shared_experts, system_config, dtype, is_moe=False) -> None:
        super().__init__()
        logging.info("Creating Decode layer {}".format(layer_id))

        self.system_config = system_config

        self.attention = MLABlock(layer_id+"_attn", hidden_size, q_lora_rank, kv_lora_rank, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, system_config, dtype)

        if is_moe:
            self.ffn = MoE(layer_id+"_moe", hidden_size, intermediate_size, num_experts_per_tok, n_experts, n_shared_experts, system_config, dtype)
        else:
            self.ffn = FFN(layer_id+"_ffn", hidden_size, intermediate_size, system_config, dtype)

    def forward(self, bsz, seqlen, ctx_len, stats):
        bsz_per_device_attn = intceil(bsz/self.system_config.dp_attn)
        self.attention.forward(bsz_per_device_attn, seqlen, ctx_len, stats=stats)

        bsz_per_device_ffn = intceil(bsz/self.system_config.dp_ffn)
        self.ffn.forward(bsz_per_device_ffn, seqlen, ctx_len, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        bsz_per_device_attn = intceil(bsz/self.system_config.dp_attn)
        mem_size = self.attention.memory_footprint(bsz_per_device_attn, ctx_len)

        bsz_per_device_ffn = intceil(bsz/self.system_config.dp_ffn)
        mem_size += self.ffn.memory_footprint(bsz_per_device_ffn)

        return mem_size # in bytes

class LMHead(Layer):
    def __init__(self, layer_id, hidden_size, vocab_size, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating LMHead layer {}".format(layer_id))

        vocab_size_per_device = intceil(vocab_size/system_config.num_nodes)
        self.head = Linear(uid=layer_id+"_head", in_features=hidden_size, out_features=vocab_size_per_device, dtype=dtype)

    def forward(self, bsz, seqlen, ctx_len, stats):
        self.head.forward(bsz, seqlen, ctx_len, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        mem_size = self.head.memory_footprint(bsz, ctx_len)
        return mem_size # in bytes
    
    def num_ops(self, bsz=None, ctx_len=None):
        n_ops = self.head.num_ops(bsz, ctx_len)
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        rw = self.head.hbm_reads(bsz, ctx_len)
        return rw # in bytes