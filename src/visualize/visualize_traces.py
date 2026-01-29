import json
import argparse

from src.core_level.common.wafer import Wafer
from src.visualize.draw_wafer import DrawWafer


ops = [
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

def visaulize_traces(args):
    mode = "decode" if args.iter.startswith("decode") else "prefill"
    with open(args.system_config, "r") as f:
        cfg = json.load(f)[mode]
    
        node_grid = cfg["node_grid"]
        core_grid = cfg["core_grid"]
    wafer = Wafer(node_grid, core_grid)

    for i, uid in enumerate(ops):
        uid = f"{args.layers}_{uid}"
        draw = DrawWafer(wafer)
        traces = wafer.load_traces(args.iter, dir_path=f"output/traces/{mode}", filter_by_uid=uid)
        draw.draw_traces(traces, out_path=f"output/visuals/{i}_{uid}.png")

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--system_config", type=str, default="configs/system.json", help="config file path for system parameters")
    argparser.add_argument("--layers", type=str, default="decode5", help="layer to generate traces for. Supports only a single layer., e.g., 'decode5'")
    argparser.add_argument("--iter", type=str, default="decode0", help="which iteration to generate traces for. e.g., 'prefill', 'decode0'")
    argparser.add_argument("--log", choices=["debug", "info", "error"], default="info", help="logging level")
    args = argparser.parse_args()

    assert len(args.layers.split(",")) == 1 and args.layers != "all", "Only single layer is supported for trace visualization."

    visaulize_traces(args)














