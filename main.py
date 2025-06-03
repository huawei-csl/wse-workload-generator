
import argparse
import json
import shutil
import os 
import sys 

from arch import build_model
from config import SystemConfig

import logging
from logger import init_logger
init_logger(level=logging.INFO)

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_config", type=str, default="configs/llama3.json")
    argparser.add_argument("--model_dtype", choices=["fp16", "fp8"], default="fp16")
    args = argparser.parse_args()

    with open(args.model_config, "r") as f:
        model_config = json.load(f) 
        print(model_config)

    out_dir = os.path.abspath("./out")
    assert out_dir not in [".", "..", "./", "/", "//"], "out_dir seems to be not safe"
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    bsz = 16
    prefill_len = 4096
    decode_len = 10

    num_nodes = 4
    for rank in range(num_nodes):
        system_config = SystemConfig(rank, num_nodes, dp_attn=2, dp_ffn=2, tp_attn=2, tp_ffn=2, pp=1, sp=1, ep=1)

        model = build_model(model_config, system_config, args.model_dtype)
        model.generate(bsz, prefill_len, decode_len)
