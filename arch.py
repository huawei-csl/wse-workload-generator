

from layers import LlamaDecodeLayer, DSv3DecodeLayer, LMHead
import logging 
from stats import RuntimeStats
from utils import intceil, divide_equal
from workload import get_moe_gate_model
from compute_graph import get_compute_graph
from tensor import reset_tensor_registry, Slice

class Model:
    def __init__(self, model_config, dist_info, dtype, out_dir) -> None:
        self.stats = RuntimeStats()
        self.moe_gate_model = None
        self.out_dir = out_dir
        self.dist_info = dist_info
        
    def new_iter(self, iter_id, bsz, seqlen):
        reset_tensor_registry()
        self.stats.new_iter(iter_id)
        if self.moe_gate_model:
            bsz_per_device = intceil(bsz/self.dist_info.dp_ffn)
            self.moe_gate_model.new_iter(iter_id, bsz_per_device, seqlen)

    '''
    Calculates memory size per device, including model weights and KV-cache. Return value is in bytes. 
    '''
    def memory_footprint(self, bsz, ctx_len):
        logging.info("Calculating memory footprint with bsz: {} and ctx_len: {}".format(bsz, ctx_len))
        self.dist_info.batch_mapping(bsz)

        memory_footprint = 0
        for l in range(len(self.layers)):
            memory_footprint += self.layers[l].memory_footprint(bsz, ctx_len)
        return memory_footprint

    def forward(self, queries, ctx_len, iter_id):
        bsz, seqlen, _ = queries.dims
        assert bsz >= self.dist_info.dp_attn, "dp_attn should not be smaller than batch size"

        self.dist_info.batch_mapping(bsz)
        self.new_iter(iter_id, bsz, seqlen)

        is_prefill = ctx_len==0
        if not is_prefill:
            assert seqlen==1, "seqlen must be 1 for decoding"

        logging.info("{} with bsz: {}, seqlen: {}, ctx_len: {}".format("Prefill" if is_prefill else "Decode", bsz, seqlen, ctx_len))

        x = queries

        batch_ids = self.dist_info.get_local_batchids("attn")
        x = Slice(x, batch_ids, axis=0).forward(self.stats)
        # x = x.slice(batch_ids, axis=0)

        if is_prefill and self.dist_info.sp > 1:
            local_seqlens = divide_equal(seqlen, self.dist_info.sp)
            start = sum(local_seqlens[:self.dist_info.rank_sp])
            end = start + local_seqlens[self.dist_info.rank_sp]
            assert end > start
            x = Slice(x, list(range(start, end)), axis=1).forward(self.stats)
            # x = x.slice(list(range(start, end)), axis=1)

        for l in range(len(self.layers)):
            x = self.layers[l].forward(x, ctx_len, self.stats)
        if self.head:
            self.head.forward(x, self.stats)

        if self.out_dir:
            if is_prefill:
                self.stats.write_to_csv(f"{self.out_dir}/prefill/node_{self.dist_info.rank}/prefill.csv")
            else:
                self.stats.write_to_csv(f"{self.out_dir}/decode/node_{self.dist_info.rank}/decode{str(iter_id-1)}.csv")
            self.stats.summarize()

        if self.out_dir:
            if is_prefill:
                get_compute_graph().dump(f"{self.out_dir}_graph/node_{self.dist_info.rank}/prefill.csv")
            else:
                get_compute_graph().dump(f"{self.out_dir}_graph/node_{self.dist_info.rank}/decode{str(iter_id-1)}.csv")

class Llama(Model):
    def __init__(self, model_config, dist_info, dtype, layer_ids, out_dir) -> None:
        super().__init__(model_config, dist_info, dtype, out_dir)

        self.dist_info = dist_info
        self.num_hidden_layers = model_config["num_hidden_layers"]
        self.hidden_size = model_config["hidden_size"]
        self.intermediate_size = model_config["intermediate_size"]
        self.num_attention_heads = model_config["num_attention_heads"]
        self.num_key_value_heads = model_config["num_key_value_heads"]
        self.vocab_size = model_config["vocab_size"]
        self.dtype = dtype

        num_layers_per_device = divide_equal(self.num_hidden_layers, dist_info.pp)[dist_info.rank_pp]

        self.layers = []
        for l in range(dist_info.rank_pp*num_layers_per_device, (dist_info.rank_pp+1)*num_layers_per_device):

            layer_id = "decode" + str(l)
            if "all" not in layer_ids and layer_id not in layer_ids:
                continue

            self.layers.append(
                LlamaDecodeLayer(
                    layer_id=layer_id, 
                    hidden_size=self.hidden_size, 
                    num_attention_heads=self.num_attention_heads, 
                    num_key_value_heads=self.num_key_value_heads,
                    intermediate_size=self.intermediate_size,
                    dist_info=dist_info,
                    dtype=dtype
                )
            )

        if "all" in layer_ids or "lm_head" in layer_ids:
            self.head = LMHead(layer_id="lm_head",
                    hidden_size=self.hidden_size,
                    vocab_size=self.vocab_size,
                    dist_info=dist_info,
                    dtype=dtype)

class DeepSeekv3(Model):
    def __init__(self, model_config, dist_info, dtype, layer_ids, out_dir) -> None:
        super().__init__(model_config, dist_info, dtype, out_dir)

        self.dist_info = dist_info
        self.num_hidden_layers = model_config["num_hidden_layers"]
        self.num_dense_layers = model_config["first_k_dense_replace"]
        self.hidden_size = model_config["hidden_size"]
        self.q_lora_rank = model_config["q_lora_rank"]
        self.kv_lora_rank = model_config["kv_lora_rank"]
        self.n_heads = model_config["num_attention_heads"]
        self.qk_nope_head_dim = model_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = model_config["qk_rope_head_dim"]
        self.v_head_dim = model_config["v_head_dim"]
        self.intermediate_size = model_config["intermediate_size"]
        self.moe_intermediate_size = model_config["moe_intermediate_size"]
        self.num_experts_per_tok = model_config["num_experts_per_tok"]
        self.n_routed_experts = model_config["n_routed_experts"]
        self.n_shared_experts = model_config["n_shared_experts"]
        self.vocab_size = model_config["vocab_size"]
        self.dtype = dtype

        num_layers_per_device = divide_equal(self.num_hidden_layers, dist_info.pp)[dist_info.rank_pp]

        self.layers = []
        moe_layer_ids = []
        for l in range(dist_info.rank_pp*num_layers_per_device, (dist_info.rank_pp+1)*num_layers_per_device):
            is_moe = l >= self.num_dense_layers 

            layer_id = "decode" + str(l)
            if "all" not in layer_ids and layer_id not in layer_ids:
                continue

            self.layers.append(
                    DSv3DecodeLayer(
                        layer_id=layer_id, 
                        hidden_size=self.hidden_size, 
                        q_lora_rank=self.q_lora_rank, 
                        kv_lora_rank=self.kv_lora_rank, 
                        n_heads=self.n_heads, 
                        qk_nope_head_dim=self.qk_nope_head_dim, 
                        qk_rope_head_dim=self.qk_rope_head_dim, 
                        v_head_dim=self.v_head_dim,
                        intermediate_size=self.intermediate_size,
                        moe_intermediate_size=self.moe_intermediate_size,
                        num_experts_per_tok=self.num_experts_per_tok,
                        n_experts=self.n_routed_experts,
                        n_shared_experts=self.n_shared_experts,
                        dist_info=dist_info,
                        dtype=dtype,
                        is_moe=is_moe
                    )
                )
            
        if "all" in layer_ids or "lm_head" in layer_ids:
            self.head = LMHead(layer_id="lm_head",
                    hidden_size=self.hidden_size,
                    vocab_size=self.vocab_size,
                    dist_info=dist_info,
                    dtype=dtype
                )
        else:
            self.head = None

        moe_layer_ids = ["decode" + str(l) + "_moe" for l in range(self.num_dense_layers, self.num_hidden_layers)]
        self.moe_gate_model = get_moe_gate_model(self.num_experts_per_tok, self.n_routed_experts, moe_layer_ids, dist_info.expert_workload_model)

    def set_global_bsz(self, global_bsz):
        for layer in self.layers:
            layer.set_global_bsz(global_bsz)

def get_arch(arch):
    if arch == "LlamaForCausalLM":
        return Llama
    elif arch == "DeepseekV3ForCausalLM":
        return DeepSeekv3
    else:
        raise NotImplementedError

def build_model(model_config, dist_info, dtype, layer_ids, out_dir):
    arch = get_arch(model_config['architectures'][0])
    return arch(model_config, dist_info, dtype, layer_ids, out_dir)

