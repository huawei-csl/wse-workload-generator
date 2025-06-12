
import json

import logging 
from logger import init_logger
from generator import Generator
from config import SystemConfig
from arch import build_model

def run(model_config, tp_attn=1, tp_ffn=1, dp_attn=1, dp_ffn=1, pp=1, ep=1):
    sp = 1
    bsz = 32
    prefill_len = 1024
    decode_len = 1024
    dtype = "fp16"

    num_nodes = tp_attn * dp_attn * pp * sp
    assert num_nodes == tp_ffn * dp_ffn * pp * ep

    generator = Generator()
    decode_cfg = SystemConfig().from_args(num_nodes, dp_attn, dp_ffn, tp_attn, tp_ffn, pp, sp, ep, expert_workload_model="uniform")

    models = []
    for rank in range(decode_cfg.num_nodes):
        models.append(build_model(model_config, decode_cfg.get_dist_info(rank), dtype, out_dir=None))

    generator.decode(models, bsz, prefill_len, decode_len, simplified_decode=True)

    total_memory_footprint, total_num_ops, total_hbm_reads = 0, 0, 0
    for i in range(len(models)):
        out = models[i].stats.summarize()
        total_memory_footprint += out[0]
        total_num_ops += out[1]
        total_hbm_reads += out[2]

    return total_memory_footprint, total_num_ops, total_hbm_reads

def test_llama():
    with open("configs/llama3.json", "r") as f:
        model_config = json.load(f) 

    init_logger(level=logging.ERROR)

    gt = run(model_config)
    
    for tp in [2,4,8]:
        total_memory_footprint, total_num_ops, total_hbm_reads = run(model_config, tp_attn=tp, tp_ffn=tp)
        assert gt[0] == total_memory_footprint
        assert gt[1] == total_num_ops
        assert gt[2] == total_hbm_reads

    for dp in [2,4,8]:
        _, total_num_ops, _ = run(model_config, dp_attn=dp, dp_ffn=dp)
        assert gt[1] == total_num_ops

    for pp in [2,4,8]:
        total_memory_footprint, total_num_ops, total_hbm_reads = run(model_config, pp=pp)
        assert gt[0] == total_memory_footprint
        assert gt[1] == total_num_ops
        assert gt[2] == total_hbm_reads

    for par in [2]:
        _, total_num_ops, _ = run(model_config, dp_attn=par, dp_ffn=par, tp_attn=par, tp_ffn=par, pp=par)
        assert gt[1] == total_num_ops

def test_deepseek():
    with open("configs/deepseekv3.json", "r") as f:
        model_config = json.load(f) 

    init_logger(level=logging.ERROR)

    gt = run(model_config)
    
    for par in [2, 4, 8, 16]:
        _, total_num_ops, _ = run(model_config, dp_attn=par, ep=par)
        assert gt[1] == total_num_ops, "{} vs {}".format(gt[1], total_num_ops)

if __name__=="__main__":
    test_llama()
    test_deepseek()