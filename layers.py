import logging

from utils import dtype_to_byte, intceil

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

class AttentionBlock(Layer):
    def __init__(self, uid, hidden_size, num_attention_heads, num_key_value_heads, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating Attention block {}".format(uid))

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
        bsz_per_device = intceil(bsz/self.system_config.dp_attn)
        for opname in self.ops:
            self.ops[opname].forward(bsz_per_device, seqlen, ctx_len, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        mem_size = sum([self.ops[opname].memory_footprint(bsz, ctx_len) for opname in self.ops])
        return mem_size # in bytes

    def num_ops(self, bsz=None, ctx_len=None):
        n_ops = sum([self.ops[opname].num_ops(bsz, ctx_len) for opname in self.ops])
        return n_ops # in terms of number of MACs
    
    def hbm_reads(self, bsz, ctx_len):
        rw = sum([self.ops[opname].hbm_reads(bsz, ctx_len) for opname in self.ops])
        return rw # in bytes

    def network_data(self, bsz):
        vecsize = sum([self.ops[opname].network_data(bsz) for opname in self.ops])
        return vecsize # in bytes
    
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

    def forward(self, bsz, seqlen, ctx_len, stats):
        bsz_per_device = intceil(bsz/self.system_config.dp_ffn)
        for opname in self.ops:
            self.ops[opname].forward(bsz_per_device, seqlen, ctx_len, stats=stats)

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
    
class DecodeLayer(Layer):
    def __init__(self, layer_id, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, system_config, dtype) -> None:
        super().__init__()
        logging.info("Creating Decode layer {}".format(layer_id))

        self.attention = AttentionBlock(layer_id+"_attn", hidden_size, num_attention_heads, num_key_value_heads, system_config, dtype)
        self.ffn = FFN(layer_id+"_ffn", hidden_size, intermediate_size, system_config, dtype)

    def forward(self, bsz, seqlen, ctx_len, stats):
        self.attention.forward(bsz, seqlen, ctx_len, stats=stats)
        self.ffn.forward(bsz, seqlen, ctx_len, stats=stats)

    def memory_footprint(self, bsz, ctx_len):
        mem_size = self.attention.memory_footprint(bsz, ctx_len) \
            + self.ffn.memory_footprint(bsz)
        return mem_size # in bytes
    
    def num_ops(self, bsz=None, ctx_len=None):
        n_ops = self.attention.num_ops(bsz, ctx_len) \
            + self.ffn.num_ops(bsz)
        return n_ops # in terms of number of MACs

    def hbm_reads(self, bsz=None, ctx_len=None):
        rw = self.attention.hbm_reads(bsz, ctx_len) \
            + self.ffn.hbm_reads()
        return rw # in bytes

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