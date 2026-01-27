import csv
import os 
import shutil
import json 

import logging
from src.node_level.common.logger import init_logger

from src.node_level.common.utils import byte_to_str

from src.core_level.common import Wafer
from src.core_level.layers import LinearLayer, GroupedLinearLayer, MLALayer, UnicastLayer, MulticastLayer, AllreduceLayer
from src.core_level.layers.view import View
from src.core_level.layers.split import Split
from src.core_level.layers.transpose import Transpose
from src.core_level.layers.concat import Concat
from src.core_level.layers.slice import Slice
from src.core_level.layers.reduce import Sum

from src.core_level.common.graph import init_graph
from src.core_level.common.tile import load_tiling_config

def generate_traces(args):
    init_logger(level=args.log.upper(), path='logs/generate_traces.log')

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

    graph = init_graph(args.iter, wafer.num_nodes, dir="output/graph")

    # clear logs
    log_dir = "logs/core_level"
    for fname in os.listdir(log_dir):
        fpath = os.path.join(log_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    
    layer_attrs = {}
    for node_id in range(wafer.num_nodes):
        logging.info("Generating traces for node {}...".format(node_id))

        csv_fname = "output/nodes/decode/node_{}/{}.csv".format(node_id, args.iter)

        # Load CSV
        data = []
        with open(csv_fname, mode="r") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=";")  # Automatically uses the first row as headers
            for row in csv_reader:
                data.append(row)

        layer_attrs[node_id] = {}
        for row in data:
            if args.layers != "all":
                if row["uid"].split("_")[0] not in layers_to_process:
                    continue

            if row["operation"] == "Linear":
                uid = row["uid"]

                in_dims = row["Dimensions"].split(" x ")[0][1:-1].split(", ")
                M, K = int(in_dims[0]), int(in_dims[1])
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                assert int(out_dims[0]) == M, "M dimension mismatch."
                M, N = int(out_dims[0]), int(out_dims[1])

                dims = (M, K, N)
                
                tile_size = load_tiling_config("configs/tiling.json", "Linear", dims, uid)

                layer_attrs[node_id][uid] = {
                    "type": LinearLayer, 
                    "attrs":{"uid": uid, "node_id": node_id, "graph": graph, "dims": dims, "tile_size": tile_size, "wafer": wafer, "prec": prec}
                }

            elif row["operation"] == "Sum":
                uid = row["uid"]
                input_dims = list(map(int, row["Dimensions"].split(" -> ")[0][1:-1].split(", ")))
                axis = int(row["Dimensions"].split(" -> ")[1])
                output_dims = list(map(int, row["Dimensions"].split(" -> ")[2][1:-1].split(", ")))

                layer_attrs[node_id][uid] = {
                    "type": Sum, 
                    "attrs":{"uid": uid, "node_id": node_id, "axis": axis, "input_dims": input_dims, "graph": graph, "wafer": wafer, "prec": prec}
                }

            elif row["operation"] == "GroupedLinear":
                uid = row["uid"]

                in_dims = row["Dimensions"].split(" x ")[0][1:-1].split(", ")
                B, M, K = int(in_dims[0]), int(in_dims[1]), int(in_dims[2])
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                assert int(out_dims[0]) == B, "Batch dimension mismatch."
                assert int(out_dims[1]) == M, "M dimension mismatch."
                B, M, N = int(out_dims[0]), int(out_dims[1]), int(out_dims[2])

                dims = (B, M, K, N)
                layer_attrs[node_id][uid] = {
                    "type": GroupedLinearLayer, 
                    "attrs":{"uid": uid, "node_id": node_id, "graph": graph, "dims": dims, "wafer": wafer, "prec": prec}
                }

            elif row["operation"] == "MLAAbsorbAttention":
                uid = row["uid"]
                out_dims = row["Dimensions"].split(" -> ")[-1][1:-1].split(", ")
                bsz, seqlen_q, num_heads, kv_lora_rank = int(out_dims[0]), int(out_dims[1]), int(out_dims[2]), int(out_dims[3])
                assert seqlen_q == 1, "Only support seqlen_q == 1 for decoding."
                pe_dims = row["Dimensions"].split(" -> ")[0].split(", PE: ")[-1][1:-1].split(", ")
                assert int(pe_dims[0]) == bsz
                _, seqlen_kv, qk_rope_head_dim = int(pe_dims[0]), int(pe_dims[1]), int(pe_dims[2])

                q_dims = bsz, seqlen_q, num_heads, kv_lora_rank
                kv_dims = bsz, seqlen_kv, kv_lora_rank
                pe_dims = bsz, seqlen_kv, qk_rope_head_dim

                q_tile_size = load_tiling_config("configs/tiling.json", "AttentionQ", q_dims, uid)
                kv_tile_size = load_tiling_config("configs/tiling.json", "AttentionKV", kv_dims, uid)

                layer_attrs[node_id][uid] = {
                    "type": MLALayer, 
                    "attrs":{
                        "uid": uid,
                        "node_id": node_id,
                        "graph": graph,
                        "q_dims": q_dims,
                        "kv_dims": kv_dims,
                        "pe_dims": pe_dims,
                        "q_tile_size": q_tile_size,
                        "kv_tile_size": kv_tile_size,
                        "wafer": wafer,
                        "prec": prec
                    }
                }
                
            elif row["operation"] == "Multicast":
                uid = row["uid"]
                dims = list(map(int, row["Dimensions"][1:-1].split(","))) 
                comm_group = row["comm. group"][1:-1].split(",")
                dst_nodes = list(map(int, comm_group))

                layer_attrs[node_id][uid] = {
                    "type": MulticastLayer, 
                    "attrs":{"uid": uid, "src": node_id, "dsts": dst_nodes, "graph": graph, "dims": dims, "wafer": wafer, "prec": prec}
                }

            elif row["operation"] == "Unicast":
                uid = row["uid"]
                dims = list(map(int, row["Dimensions"][1:-1].split(", ")))
                dst_node = int(row["comm. group"])

                layer_attrs[node_id][uid] = {
                    "type": UnicastLayer, 
                    "attrs":{"uid": uid, "node_id": node_id, "src": node_id, "dst": dst_node, "graph": graph, "dims": dims, "wafer": wafer, "prec": prec}
                }

            elif row["operation"] == "AllReduce":
                uid = row["uid"]
                dims = list(map(int, row["Dimensions"][1:-1].split(", ")))
                comm_group = list(map(int, row["comm. group"][1:-1].split(",")))

                layer_attrs[node_id][uid] = {
                    "type": AllreduceLayer, 
                    "attrs":{"uid": uid, "node_id": node_id, "comm_group": comm_group, "graph": graph, "dims": dims, "wafer": wafer, "prec": prec}
                }

            elif row["operation"] == "View":
                uid = row["uid"]
                input_dims = list(map(int, row["Dimensions"].split(" -> ")[0][1:-1].split(", ")))
                output_dims = list(map(int, row["Dimensions"].split(" -> ")[1][1:-1].split(", ")))

                layer_attrs[node_id][uid] = {
                    "type": View, 
                    "attrs":{"uid": uid, "node_id": node_id, "input_dims": input_dims, "output_dims": output_dims, "graph": graph, "prec": prec}
                }

            elif row["operation"] == "Split":
                uid = row["uid"]
                axis = int(row["Dimensions"].split(" -> ")[0])
                split_dims = list(map(int, row["Dimensions"].split(" -> ")[1][1:-1].split(", ")))
                input_dims = list(map(int, row["Dimensions"].split(" -> ")[2][1:-1].split(", ")))

                layer_attrs[node_id][uid] = {
                    "type": Split, 
                    "attrs":{"uid": uid, "node_id": node_id, "axis": axis, "split_dims": split_dims, "input_dims": input_dims, "graph": graph, "prec": prec}
                }

            elif row["operation"] == "Transpose":
                uid = row["uid"]

                axes = list(map(int, row["Dimensions"].split(" -> ")[0][1:-1].split(", ")))
                input_dims = list(map(int, row["Dimensions"].split(" -> ")[1][1:-1].split(", ")))
                output_dims = list(map(int, row["Dimensions"].split(" -> ")[2][1:-1].split(", ")))

                layer_attrs[node_id][uid] = {
                    "type": Transpose, 
                    "attrs":{"uid": uid, "node_id": node_id, "axes": axes, "input_dims": input_dims, "output_dims": output_dims, "graph": graph, "prec": prec}
                }
            
            elif row["operation"] == "Concat":
                uid = row["uid"]

                axis = int(row["Dimensions"].split(" -> ")[0])
                concat_dims = list(map(int, row["Dimensions"].split(" -> ")[1][1:-1].split(", ")))
                output_dims = list(map(int, row["Dimensions"].split(" -> ")[2][1:-1].split(", ")))

                input_dims = []
                for d in concat_dims:
                    input_dim = list(output_dims)
                    input_dim[axis] = d
                    input_dims.append(input_dim)

                layer_attrs[node_id][uid] = {
                    "type": Concat, 
                    "attrs":{"uid": uid, "node_id": node_id, "axis": axis, "input_dims": input_dims, "graph": graph, "prec": prec}
                }

            elif row["operation"] == "Slice":
                uid = row["uid"]

                input_dims = list(map(int, row["Dimensions"].split(" -> ")[0][1:-1].split(", ")))
                axis = int(row["Dimensions"].split(" -> ")[1])
                rng_start, rng_end = map(int, row["Dimensions"].split(" -> ")[2].split(":"))
                index_rng = (rng_start, rng_end)

                layer_attrs[node_id][uid] = {
                    "type": Slice, 
                    "attrs":{"uid": uid, "node_id": node_id, "axis": axis, "index_rng": index_rng, "input_dims": input_dims, "graph": graph, "prec": prec}
                }

            else:
                # raise NotImplementedError
                logging.warning("Operation {} not recognized.".format(row["operation"]))

    # Run layers from the compute graph in a BFS order
    topo_queue = graph.get_topological_order()

    for node in topo_queue:
        print(f"Mapping {node.uid} in node_id {node.node_id}")

        if node.op_type in ["root"]:
            continue

        node_id = node.node_id
        uid = node.uid

        if uid not in layer_attrs[node_id]:
            continue

        layer_class = layer_attrs[node_id][uid]["type"]
        layer_params = layer_attrs[node_id][uid]["attrs"]

        layer = layer_class(**layer_params)
        layer.log_stats()

    wafer.export_traces(args.iter, "output/traces/decode")

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