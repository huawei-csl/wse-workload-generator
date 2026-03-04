import json
import argparse
import pandas as pd

from copy import deepcopy

from src.core_level.common.wafer import Wafer
from src.visualize.draw_wafer import DrawWafer

ops_multicast = [
    "attn_absorb_wkva",
    "attn_absorb_wqa",
    "attn_absorb_wqb",
    "attn_absorb_wkvb1",
    "attn_absorb_absorbattn",
    "attn_absorb_ar_sp",
    "attn_absorb_wkvb2",
    "attn_absorb_wo",
    "attn_absorb_ar_tp",
    "moe_gate",
    "moe_multicast_exp",
    "moe_exp",
    "moe_exp_shared",
    "moe_unicast",
    "moe_sum",
    "moe_multicast_dp",
]

ops_alltoall = [
    "attn_absorb_wkva",
    "attn_absorb_wqa",
    "attn_absorb_wqb",
    "attn_absorb_wkvb1",
    "attn_absorb_absorbattn",
    "attn_absorb_ar_sp",
    "attn_absorb_wkvb2",
    "attn_absorb_wo",
    "attn_absorb_ar_tp",
    "moe_gate",
    "moe_a2a_disp_unicast",
    "moe_exp",
    "moe_exp_shared",
    "moe_combine_a2a_unicast",
    "moe_sum",
    "moe_multicast_dp",
]

def visaulize_traces(args):
    mode = "decode" if args.iter.startswith("decode") else "prefill"
    with open(args.system_config, "r") as f:
        cfg = json.load(f)[mode]
    
        node_grid = cfg["node_grid"]
        core_grid = cfg["core_grid"]
    wafer = Wafer(node_grid, core_grid)

    if cfg["moe_comm"] == "multicast":
        ops = ops_multicast
    elif cfg["moe_comm"] == "alltoall":
        ops = ops_alltoall
    else:
        raise ValueError(f"Ops for this communication type not defined: {cfg['moe_comm']}")

    comm_matrices = []
    for i, uid in enumerate(ops):
        uid = f"{args.layers}_{uid}"
        draw = DrawWafer(wafer)
        traces = wafer.load_traces(args.iter, dir_path=f"{args.outdir}/traces/{mode}", filter_by_uid=uid)
        # wafer.analyze_traces(traces)
        traffic = wafer.extract_traffic(traces)

        draw.draw_traces(traffic, out_path=f"{args.outdir}/visuals/{i}_{uid}.png")

        comm_matrix = wafer.calc_comm_matrix(traffic, f"{args.outdir}/comm_matrices/{i}_{uid}.csv")
        comm_matrices.append(comm_matrix)

        draw.draw_comm_matrix_by_core(comm_matrix, out_path=f"{args.outdir}/comm_matrices/{i}_{uid}_core.png")
        draw.draw_comm_matrix_by_node(comm_matrix, out_path=f"{args.outdir}/comm_matrices/{i}_{uid}_node.png")

    total_comm = deepcopy(comm_matrices[0])
    for comm_matrix in comm_matrices[1:]:
        for src in comm_matrix:
            for dst in comm_matrix[src]:
                total_comm[src][dst] = total_comm[src][dst] + comm_matrix[src][dst]

    draw.draw_comm_matrix_by_core(total_comm, out_path=f"{args.outdir}/comm_matrices/total_{uid}_core.png")
    draw.draw_comm_matrix_by_node(total_comm, out_path=f"{args.outdir}/comm_matrices/total_{uid}_node.png")

    total_comm_df = pd.DataFrame(total_comm)
    total_comm_df.to_csv(f"{args.outdir}/comm_matrices/total_{uid}.csv")

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--system_config", type=str, default="configs/system.json", help="config file path for system parameters")
    argparser.add_argument("--layers", type=str, default="decode5", help="layer to generate traces for. Supports only a single layer., e.g., 'decode5'")
    argparser.add_argument("--iter", type=str, default="decode0", help="which iteration to generate traces for. e.g., 'prefill', 'decode0'")
    argparser.add_argument("--log", choices=["debug", "info", "error"], default="info", help="logging level")
    argparser.add_argument("--outdir", type=str, default="./output", help="directory for generated files")

    args = argparser.parse_args()

    assert len(args.layers.split(",")) == 1 and args.layers != "all", "Only single layer is supported for trace visualization."

    visaulize_traces(args)
