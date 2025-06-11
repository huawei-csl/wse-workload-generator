
import argparse
import json
import shutil
import os 
import sys 
from copy import deepcopy
from arch import build_model
from config import SystemConfig
from generator import Generator

import logging
from logger import init_logger
init_logger(level=logging.INFO)

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_config", type=str, default="configs/deepseekv3.json", help="HF json file path for model config")
    argparser.add_argument("--system_config", type=str, default="configs/system.json", help="config file path for system parameters")
    argparser.add_argument("--bsz", type=int, default=1, help="batch size")
    argparser.add_argument("--prefill_len", type=int, default=32, help="prefill length")
    argparser.add_argument("--decode_len", type=int, default=10, help="decode length")
    argparser.add_argument("--only_decode", type=int, choices=[0,1], default=1, help="1: skips prefill, 0: run both prefill and decode")
    argparser.add_argument("--simplified_decode", type=int, choices=[0,1], default=1, help="0: full decode run, 1: run only first and last decode iterations, rest can be interpolated")
    argparser.add_argument("--dtype", choices=["fp16", "fp8"], default="fp16", help="numeric precision")
    args = argparser.parse_args()

    with open(args.model_config, "r") as f:
        model_config = json.load(f) 
        print(model_config)

    out_dir = os.path.abspath("./out")
    assert out_dir not in [".", "..", "./", "/", "//"], "out_dir seems to be not safe"
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    generator = Generator()
    if not args.only_decode:
        prefill_cfg = SystemConfig(args.system_config, mode="prefill")
        prefill_models = []
        for rank in range(prefill_cfg.num_nodes):
            prefill_models.append(build_model(model_config, deepcopy(prefill_cfg), rank, args.dtype))
        generator.prefill(prefill_models, args.bsz, args.prefill_len)

    decode_cfg = SystemConfig(args.system_config, mode="decode")
    decode_models = []
    for rank in range(decode_cfg.num_nodes):
        decode_models.append(build_model(model_config, deepcopy(decode_cfg), rank, args.dtype))
    generator.decode(decode_models, args.bsz, args.prefill_len, args.decode_len, args.simplified_decode)
