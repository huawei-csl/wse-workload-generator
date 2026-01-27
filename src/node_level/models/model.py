import logging 
from src.node_level.common.stats import NodeStats
from src.node_level.common.utils import intceil, divide_equal

from src.node_level.common.compute_graph import get_compute_graph
from src.node_level.common.tensor import reset_tensor_registry, Slice

class Model:
    def __init__(self, model_config, dist_info, dtype, out_dir) -> None:
        self.stats = NodeStats()
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

        if is_prefill and self.dist_info.sp > 1:
            local_seqlens = divide_equal(seqlen, self.dist_info.sp)
            start = sum(local_seqlens[:self.dist_info.rank_sp])
            end = start + local_seqlens[self.dist_info.rank_sp]
            assert end > start
            x = Slice(x, list(range(start, end)), axis=1).forward(self.stats)

        for l in range(len(self.layers)):
            x = self.layers[l].forward(x, ctx_len, self.stats)
        if self.head:
            self.head.forward(x, self.stats)

        if self.out_dir:
            if is_prefill:
                self.stats.write_to_csv(f"{self.out_dir}/nodes/prefill/node_{self.dist_info.rank}/prefill.csv")
            else:
                self.stats.write_to_csv(f"{self.out_dir}/nodes/decode/node_{self.dist_info.rank}/decode{str(iter_id-1)}.csv")
            self.stats.summarize()

        if self.out_dir:
            if is_prefill:
                get_compute_graph().dump(f"{self.out_dir}/graph/node_{self.dist_info.rank}/prefill.csv")
            else:
                get_compute_graph().dump(f"{self.out_dir}/graph/node_{self.dist_info.rank}/decode{str(iter_id-1)}.csv")

def get_arch(arch):
    if arch == "LlamaForCausalLM":
        from src.node_level.models.llama import Llama
        return Llama
    elif arch == "DeepseekV3ForCausalLM":
        from src.node_level.models.dsv3 import DeepSeekv3
        return DeepSeekv3
    else:
        raise NotImplementedError

def build_model(model_config, dist_info, dtype, layer_ids, out_dir):
    arch = get_arch(model_config['architectures'][0])
    return arch(model_config, dist_info, dtype, layer_ids, out_dir)

