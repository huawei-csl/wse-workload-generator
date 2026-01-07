import csv
import os 
import shutil
import json 

import logging
from logger import init_logger

from core_level.common import Wafer
from core_level.layers import LinearLayer, GroupedLinearLayer, MLALayer, UnicastLayer, MulticastLayer, AllreduceLayer
from core_level.layers.view import View
from core_level.common.graph import init_graph
from core_level.common.tile import load_tiling_config

def generate_traces(args):
    init_logger(level=args.log.upper())

    curr_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(curr_directory, "../traces")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    if args.layers != "all":
        layers_to_process = args.layers.split(",")
    
    mode = "decode" if args.iter.startswith("decode") else "prefill"
    with open(args.system_config, "r") as f:
        cfg = json.load(f)[mode]
    
        node_grid = cfg["node_grid"]
        core_grid = cfg["core_grid"]

    prec = args.dtype

    wafer = Wafer(node_grid, core_grid)

    graph = init_graph(args.iter, wafer.num_nodes)

    for node_id in range(wafer.num_nodes):
        logging.info("Generating traces for node {}...".format(node_id))

        csv_fname = "out/decode/node_{}/{}.csv".format(node_id, args.iter)

        # Load CSV
        data = []
        with open(csv_fname, mode="r") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=";")  # Automatically uses the first row as headers
            for row in csv_reader:
                data.append(row)

        # Print loaded data for debugging
        for row in data:
            if args.layers != "all":
                if row["uid"].split("_")[0] not in layers_to_process:
                    continue

            if row["operation"] == "Linear":
                layer_uid = row["uid"]

                in_dims = row["Dimensions"].split(" x ")[0][1:-1].split(", ")
                M, K = int(in_dims[0]), int(in_dims[1])
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                assert int(out_dims[0]) == M, "M dimension mismatch."
                M, N = int(out_dims[0]), int(out_dims[1])

                dims = (M, K, N)
                
                tile_size = load_tiling_config("configs/tiling.json", "Linear", dims)

                LinearLayer(layer_uid, node_id, graph, dims, tile_size, wafer, prec)

            elif row["operation"] == "GroupedLinear":
                layer_uid = row["uid"]

                in_dims = row["Dimensions"].split(" x ")[0][1:-1].split(", ")
                B, M, K = int(in_dims[0]), int(in_dims[1]), int(in_dims[2])
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                assert int(out_dims[0]) == B, "Batch dimension mismatch."
                assert int(out_dims[1]) == M, "M dimension mismatch."
                B, M, N = int(out_dims[0]), int(out_dims[1]), int(out_dims[2])

                dims = (B, M, K, N)
                GroupedLinearLayer(layer_uid, node_id, graph, dims, wafer, prec)

            elif row["operation"] == "MLAAbsorbAttention":
                layer_id = row["uid"]
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                bsz, seqlen_q, num_heads, kv_lora_rank = int(out_dims[0]), int(out_dims[1]), int(out_dims[2]), int(out_dims[3])
                assert seqlen_q == 1, "Only support seqlen_q == 1 for decoding."
                pe_dims = row["Dimensions"].split(" -> ")[0].split(", PE: ")[-1][1:-1].split(", ")
                assert int(pe_dims[0]) == bsz
                _, seqlen_kv, qk_rope_head_dim = int(pe_dims[0]), int(pe_dims[1]), int(pe_dims[2])

                q_dims = bsz, seqlen_q, num_heads, kv_lora_rank
                kv_dims = bsz, seqlen_kv, kv_lora_rank
                pe_dims = bsz, seqlen_kv, qk_rope_head_dim

                MLALayer(layer_id, node_id, graph, q_dims, kv_dims, pe_dims, wafer, prec)

            elif row["operation"] == "Multicast":
                layer_id = row["uid"]
                dims = row["Dimensions"][1:-1].split(",")
                dims = [int(dims[0]),]
                comm_group = row["comm. group"][1:-1].split(",")
                dst_nodes = list(map(int, comm_group))

                MulticastLayer(layer_id, node_id, dst_nodes, graph, dims, wafer, prec)

            elif row["operation"] == "Unicast":
                layer_id = row["uid"]
                dims = row["Dimensions"][1:-1].split(",")
                assert len(dims) == 1
                dims = [int(dims[0]),]
                dst_node = int(row["comm. group"])

                UnicastLayer(layer_id, node_id, dst_node, graph, dims, wafer, prec)

            elif row["operation"] == "AllReduce":
                layer_id = row["uid"]
                dims = list(map(int, row["Dimensions"][1:-1].split(", ")))
                comm_group = list(map(int, row["comm. group"][1:-1].split(",")))

                AllreduceLayer(layer_id, node_id, comm_group, graph, dims, wafer, prec)

            elif row["operation"] == "AlltoAll":
                layer_id = row["uid"]
                dims = list(map(int, row["Dimensions"][1:-1].split(", ")))

                comm_group = list(map(int, row["comm. group"][1:-1].split(",")))
                dst_nodes = list(map(int, comm_group))

                # model all-to-all as multicast from each node to all nodes
                MulticastLayer(layer_id, node_id, dst_nodes, graph, dims, wafer, prec)

            elif row["operation"] == "View":
                layer_id = row["uid"]
                input_dims = list(map(int, row["Dimensions"].split("->")[0][1:-1].split(", ")))
                output_dims = list(map(int, row["Dimensions"].split("->")[1][1:-1].split(", ")))

                View(layer_id, node_id, input_dims, output_dims, graph, prec)
            else:
                pass
                # raise NotImplementedError("Operation {} not implemented.".format(row["operation"])) 

    wafer.export_traces(args.iter, "traces/decode")

import argparse
if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--system_config", type=str, default="configs/system.json", help="config file path for system parameters")
    argparser.add_argument("--layers", type=str, default="all", help="layers to generate traces for, e.g., 'all', 'decode0,decode1,decode2'")
    argparser.add_argument("--iter", type=str, default="decode0", help="which iteration to generate traces for. e.g., 'prefill', 'decode0'")
    argparser.add_argument("--dtype", choices=["fp16", "fp8"], default="fp16", help="numeric precision")
    argparser.add_argument("--log", choices=["debug", "info", "error"], default="info", help="logging level")
    args = argparser.parse_args()
    generate_traces(args)