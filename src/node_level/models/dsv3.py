
from src.node_level.layers.decode import DSv3DecodeLayer
from src.node_level.layers.lmhead import LMHead
from src.node_level.models.model import Model

from src.node_level.common.workload import get_moe_gate_model
from src.node_level.common.utils import divide_equal

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

        assert self.n_routed_experts >= self.dist_info.ep, "n_routed_experts can not be smaller than expert parallelism"

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
            # Consider head as a dense FFN layer
            self.head = LMHead(
                layer_id="lm_head",
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                dist_info=dist_info,
                dtype=dtype,
            )
        else:
            self.head = None

        if len(self.layers) == 0 and self.head is None:
            assert False, "No layers selected for the model."

        moe_layer_ids = ["decode" + str(l) + "_moe" for l in range(self.num_dense_layers, self.num_hidden_layers)]
        self.moe_gate_model = get_moe_gate_model(self.num_experts_per_tok, self.n_routed_experts, moe_layer_ids, dist_info.expert_workload_model)

    def set_global_bsz(self, global_bsz):
        for layer in self.layers:
            layer.set_global_bsz(global_bsz)