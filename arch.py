

from layers import LlamaDecodeLayer, DSv3DecodeLayer, LMHead
import logging 
from stats import RuntimeStats
from utils import intceil, divide_equal
from workload import get_moe_gate_model

class Model:
    def __init__(self, model_config, dist_info, dtype, out_dir) -> None:
        self.stats = RuntimeStats()
        self.moe_gate_model = None
        self.out_dir = out_dir

    def new_iter(self, iter_id, bsz, seqlen):
        self.stats.new_iter(iter_id)
        if self.moe_gate_model:
            self.moe_gate_model.new_iter(iter_id, bsz, seqlen)

    '''
    Calculates memory size per device, including model weights and KV-cache. Return value is in bytes. 
    '''
    def memory_footprint(self, bsz, ctx_len):
        logging.info("Calculating memory footprint with bsz: {} and ctx_len: {}".format(bsz, ctx_len))
        memory_footprint = 0
        for l in range(len(self.layers)):
            memory_footprint += self.layers[l].memory_footprint(bsz, ctx_len)
        return memory_footprint

    def forward(self, bsz, seqlen, ctx_len, iter_id):
        self.new_iter(iter_id, bsz, seqlen)

        is_prefill = ctx_len==0
        if not is_prefill:
            assert seqlen==1, "seqlen must be 1 for decoding"

        logging.info("{} with bsz: {}, seqlen: {}, ctx_len: {}".format("Prefill" if is_prefill else "Decode", bsz, seqlen, ctx_len))

        for l in range(len(self.layers)):
            self.layers[l].forward(bsz, seqlen, ctx_len, self.stats)
        self.head.forward(bsz*seqlen, self.stats)

        if self.out_dir:
            if is_prefill:
                self.stats.write_to_csv(f"{self.out_dir}/prefill/node_{self.dist_info.rank}/prefill.csv")
            else:
                self.stats.write_to_csv(f"{self.out_dir}/decode/node_{self.dist_info.rank}/decode{str(iter_id-1)}.csv")
            self.stats.summarize()

class Llama(Model):
    def __init__(self, model_config, dist_info, dtype, out_dir) -> None:
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
            self.layers.append(
                LlamaDecodeLayer(
                    layer_id="decode" + str(l), 
                    hidden_size=self.hidden_size, 
                    num_attention_heads=self.num_attention_heads, 
                    num_key_value_heads=self.num_key_value_heads,
                    intermediate_size=self.intermediate_size,
                    dist_info=dist_info,
                    dtype=dtype
                )
            )
        self.head = LMHead(layer_id="lm_head",
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                dist_info=dist_info,
                dtype=dtype)

class DeepSeekv3(Model):
    def __init__(self, model_config, dist_info, dtype, out_dir) -> None:
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

            self.layers.append(
                    DSv3DecodeLayer(
                        layer_id="decode" + str(l), 
                        hidden_size=self.hidden_size, 
                        q_lora_rank=self.q_lora_rank, 
                        kv_lora_rank=self.kv_lora_rank, 
                        n_heads=self.n_heads, 
                        qk_nope_head_dim=self.qk_nope_head_dim, 
                        qk_rope_head_dim=self.qk_rope_head_dim, 
                        v_head_dim=self.v_head_dim,
                        intermediate_size=self.intermediate_size,
                        num_experts_per_tok=self.num_experts_per_tok,
                        n_experts=self.n_routed_experts,
                        n_shared_experts=self.n_shared_experts,
                        dist_info=dist_info,
                        dtype=dtype,
                        is_moe=is_moe
                    )
                )
            
            if is_moe:
                moe_layer_ids.append("decode" + str(l) + "_moe")

        self.head = LMHead(layer_id="lm_head",
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                dist_info=dist_info,
                dtype=dtype
            )

        self.moe_gate_model = get_moe_gate_model(self.num_experts_per_tok, self.n_routed_experts, moe_layer_ids, dist_info.expert_workload_model)

def get_arch(arch):
    if arch == "LlamaForCausalLM":
        return Llama
    elif arch == "DeepseekV3ForCausalLM":
        return DeepSeekv3
    else:
        raise NotImplementedError

def build_model(model_config, dist_info, dtype, out_dir):
    arch = get_arch(model_config['architectures'][0])
    return arch(model_config, dist_info, dtype, out_dir)

