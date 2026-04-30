
from src.node_level.layers.decode import LlamaDecodeLayer
from src.node_level.layers.lmhead import LMHead
from src.node_level.models.model import Model
from src.node_level.common.utils import divide_equal

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

        layer_id_set = set(layer_ids.split(","))

        self.layers = []
        for l in range(dist_info.rank_pp*num_layers_per_device, (dist_info.rank_pp+1)*num_layers_per_device):

            layer_id = "decode" + str(l)
            if "all" not in layer_id_set and layer_id not in layer_id_set:
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

        if "all" in layer_id_set or "lm_head" in layer_id_set:
            self.head = LMHead(layer_id="lm_head",
                    hidden_size=self.hidden_size,
                    vocab_size=self.vocab_size,
                    dist_info=dist_info,
                    dtype=dtype)
        else:
            self.head = None

        if len(self.layers) == 0 and self.head is None:
            assert False, "No layers selected for the model."

    def set_global_bsz(self, global_bsz):
        for layer in self.layers:
            layer.set_global_bsz(global_bsz)