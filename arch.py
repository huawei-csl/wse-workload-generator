

from layers import DecodeLayer, LMHead
import logging 
from stats import RuntimeStats
from utils import intceil 

class Model:
    def __init__(self) -> None:
        pass

class Llama(Model):
    def __init__(self, model_config, system_config, dtype) -> None:
        super().__init__()

        self.rank = system_config.rank
        self.stats = RuntimeStats()

        self.num_hidden_layers = model_config["num_hidden_layers"]
        self.hidden_size = model_config["hidden_size"]
        self.intermediate_size = model_config["intermediate_size"]
        self.num_attention_heads = model_config["num_attention_heads"]
        self.num_key_value_heads = model_config["num_key_value_heads"]
        self.vocab_size = model_config["vocab_size"]

        self.dtype = dtype

        num_layers_per_device = intceil(self.num_hidden_layers/system_config.pp)

        self.layers = []
        for l in range(system_config.rank_pp*num_layers_per_device, (system_config.rank_pp+1)*num_layers_per_device):
            self.layers.append(
                DecodeLayer(
                    layer_id="decode" + str(l), 
                    hidden_size=self.hidden_size, 
                    num_attention_heads=self.num_attention_heads, 
                    num_key_value_heads=self.num_key_value_heads,
                    intermediate_size=self.intermediate_size,
                    system_config=system_config,
                    dtype=dtype
                )
            )
        self.layers.append(
            LMHead(layer_id="lm_head",
                   hidden_size=self.hidden_size,
                   vocab_size=self.vocab_size,
                   system_config=system_config,
                   dtype=dtype)
        )

    def new_iter(self):
        self.stats.new_iter()

    '''
    Calculates memory size per device, including model weights and KV-cache. Return value is in bytes. 
    '''
    def memory_footprint(self, bsz, ctx_len):
        logging.info("Calculating memory footprint with bsz: {} and ctx_len: {}".format(bsz, ctx_len))
        memory_footprint = 0
        for l in range(len(self.layers)):
            memory_footprint += self.layers[l].memory_footprint(bsz, ctx_len)
        return memory_footprint

    def forward(self, bsz, seqlen, ctx_len):
        is_prefill = ctx_len==0
        if not is_prefill:
            assert seqlen==1, "seqlen must be 1 for decoding"

        logging.info("{} with bsz: {}, seqlen: {}, ctx_len: {}".format("Prefill" if is_prefill else "Decode", bsz, seqlen, ctx_len))

        for l in range(len(self.layers)):
            self.layers[l].forward(bsz, seqlen, ctx_len, self.stats)

    def generate(self, bsz, prefill_len, decode_len):
        self.new_iter()
        self.forward(bsz, seqlen=prefill_len, ctx_len=0)

        self.stats.write_to_csv(f"out/node_{self.rank}/prefill.csv")
        self.stats.summarize()

        for i in range(decode_len):
            self.new_iter()
            ctx_len = prefill_len + i
            self.forward(bsz, seqlen=1, ctx_len=ctx_len)

            self.stats.write_to_csv(f"out/node_{self.rank}/decode_{i}.csv")
            self.stats.summarize()


    '''
    Calculates number of MACs per device. 
    '''
    def num_ops(self, bsz, ctx_len):
        logging.info("Calculating number of MACs with bsz: {} and ctx_len: {}".format(bsz, ctx_len))
        n_ops = 0
        for l in range(len(self.layers)):
            n_ops += self.layers[l].num_ops(bsz, ctx_len)
        return n_ops

    '''
    Calculates HBM reads in terms of bytes. 
    '''
    def hbm_reads(self, bsz, ctx_len):
        logging.info("Calculating HBM reads with bsz: {} and ctx_len: {}".format(bsz, ctx_len))
        rw = 0
        for l in range(len(self.layers)):
            rw += self.layers[l].hbm_reads(bsz, ctx_len)
        return rw
    
def get_arch(arch):
    if arch == "LlamaForCausalLM":
        return Llama
    else:
        raise NotImplementedError

def build_model(model_config, system_config, dtype):
    arch = get_arch(model_config['architectures'][0])
    return arch(model_config, system_config, dtype)

