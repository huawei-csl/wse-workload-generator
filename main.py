
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

def nodes_to_simulate(nodes, max_num_nodes):
    # Convert args.nodes to a list of integers
    if args.nodes == "all":
        return list(range(max_num_nodes))
    else:
        nodes = []
        for part in args.nodes.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                nodes.extend(range(start, end + 1))
            else:
                nodes.append(int(part))
    return nodes

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_config", type=str, default="configs/deepseekv3.json", help="HF json file path for model config")
    argparser.add_argument("--system_config", type=str, default="configs/system.json", help="config file path for system parameters")
    argparser.add_argument("--bsz", type=int, default=1, help="batch size")
    argparser.add_argument("--prefill_len", type=int, default=32, help="prefill length")
    argparser.add_argument("--decode_len", type=int, default=10, help="decode length")
    argparser.add_argument("--only_decode", type=int, choices=[0,1], default=1, help="1: skips prefill, 0: run both prefill and decode")
    argparser.add_argument("--simplified_decode", type=int, choices=[0,1], default=1, help="0: full decode run, 1: run only first and last decode iterations, rest can be interpolated")
    argparser.add_argument("--nodes", type=str, default="all", help="nodes to run, e.g., 'all', '0,1,2', '0-3' (for 4 nodes), '0-3,5' (for 5 nodes)")
    argparser.add_argument("--dtype", choices=["fp16", "fp8"], default="fp16", help="numeric precision")
    argparser.add_argument("--log", choices=["debug", "info", "error"], default="info", help="logging level")
    argparser.add_argument("--outdir", type=str, default="./out", help="directory for generated csv files")
    args = argparser.parse_args()

    with open(args.model_config, "r") as f:
        model_config = json.load(f) 
        print(model_config)

    init_logger(level=args.log.upper())

    out_dir = os.path.abspath(args.outdir)
    assert out_dir not in [".", "..", "./", "/", "//"], "out_dir seems to be not safe"
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    generator = Generator()
    if not args.only_decode:
        prefill_cfg = SystemConfig().from_json(args.system_config, mode="prefill")
        nodes = nodes_to_simulate(args.nodes, prefill_cfg.num_nodes)

        prefill_models = []
        for rank in nodes:
            prefill_models.append(build_model(model_config, prefill_cfg.get_dist_info(rank), args.dtype, out_dir))

        generator.prefill(prefill_models, args.bsz, args.prefill_len)

    decode_cfg = SystemConfig().from_json(args.system_config, mode="decode")
    nodes = nodes_to_simulate(args.nodes, decode_cfg.num_nodes)

    decode_models = []
    for rank in nodes:
        decode_models.append(build_model(model_config, decode_cfg.get_dist_info(rank), args.dtype, out_dir))
    generator.decode(decode_models, args.bsz, args.prefill_len, args.decode_len, args.simplified_decode)
