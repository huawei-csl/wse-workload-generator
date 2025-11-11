
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_wafer(core_grid, node_grid):
    full_grid = [np.array(node_grid[0])*core_grid[0], np.array(node_grid[1])*core_grid[1]]

    wafer = np.ones(full_grid)

    # Plot the board
    plt.imshow(wafer, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure squares are equal
    
    plt.grid(color="black", linestyle="-", linewidth=1)
    plt.gca().set_xticks(np.arange(-0.5, full_grid[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, full_grid[0], 1), minor=True)
    plt.gca().tick_params(which="minor", size=0)  # Hide minor ticks
    plt.gca().grid(which="minor", color="black", linestyle="-", linewidth=1)

    # Draw the node grid
    for row in range(node_grid[0]):  # Iterate over rows
        for col in range(node_grid[1]):  # Iterate over columns
            # Calculate the bottom-left corner of each rectangle
            x = col * core_grid[1] - 0.5
            y = row * core_grid[0] - 0.5
            
            # Create a rectangle
            rect = patches.Rectangle((x, y), core_grid[1], core_grid[0], 
                                    linewidth=2, edgecolor='black', facecolor='none')
            plt.gca().add_patch(rect)

def draw_arrow(src, dst, color="red", alpha=0.1):
    # Draw the arrow
    plt.annotate(
        "",  # No text for the annotation
        xy=(dst[1], dst[0]),  # Destination (x, y)
        xytext=(src[1], src[0]),  # Source (x, y)
        arrowprops=dict(arrowstyle="-", color=color, lw=1, alpha=min(alpha, 1.0)),  # Arrow style
    )

def get_num_hops(src, dst):
    dist = abs(src[0] - dst[0]) + abs(src[1] - dst[1])
    dist += 1 # assume a local read/write costs one hop
    return dist

def register_move(stats_byte, src, dst, size):
    key = (tuple(src), tuple(dst))
    if key not in stats_byte:
        stats_byte[key] = 0
    stats_byte[key] += size


def get_2d_coord(core_id, core_grid):
    assert core_id < core_grid[0] * core_grid[1]

    row = core_id // core_grid[1]
    col = core_id % core_grid[1]
    return [row, col]

def map_with_quadrants(node_id, quad_grid, node_grid):
    assert node_grid[0] % quad_grid[0] == 0
    assert node_grid[1] % quad_grid[1] == 0
    
    num_nodes = node_grid[0] * node_grid[1]
    assert node_id < num_nodes

    quad_dims = [node_grid[0] // quad_grid[0], node_grid[1] // quad_grid[1]]
    quad_size = quad_dims[0] * quad_dims[1]

    quadrant = ((node_id//quad_size)//quad_grid[1], (node_id//quad_size)%quad_grid[1])
    sub_coords = ((node_id%quad_size)//quad_dims[1], (node_id%quad_size)%quad_dims[1])

    coords = (sub_coords[0] + quadrant[0]*quad_dims[0], sub_coords[1] + quadrant[1]*quad_dims[1])
    return coords

# node_grid = [4, 6]
# quad_grid = [2, 2]

# num_nodes = node_grid[0] * node_grid[1]
# for node_id in range(num_nodes):
#     coords = map_with_quadrants(node_id, quad_grid, node_grid)

# for r in range(node_grid[0]):
#     for c in range(node_grid[1]):
#         ind = coords.index((r,c))
#         print(f"{ind:2} ", end="")
#     print("")

# exit()

def get_glob_coord(core_id, core_grid, node_grid):
    num_cores_per_node = core_grid[0] * core_grid[1]
    node_id = core_id // num_cores_per_node
    # node_r, node_c = get_2d_coord(node_id, node_grid)
    node_r, node_c = map_with_quadrants(node_id, [2,2], node_grid)

    local_core_id = core_id % num_cores_per_node
    local_core_r, local_core_c = get_2d_coord(local_core_id, core_grid)

    glob_r = node_r * core_grid[0] + local_core_r
    glob_c = node_c * core_grid[1] + local_core_c

    return [glob_r, glob_c]


import json
def get_expected_vals(bsz, seqlen_kv, model_config):
    with open(model_config, "r") as f:
        model_config = json.load(f) 

    hidden_size = model_config["hidden_size"]
    q_lora_rank = model_config["q_lora_rank"]
    kv_lora_rank = model_config["kv_lora_rank"]
    num_attention_heads = model_config["num_attention_heads"]
    qk_nope_head_dim = model_config["qk_nope_head_dim"]
    qk_rope_head_dim = model_config["qk_rope_head_dim"]
    v_head_dim = model_config["v_head_dim"]
    moe_intermediate_size = model_config["moe_intermediate_size"] 
    n_routed_experts = model_config["n_routed_experts"]
    n_shared_experts = model_config["n_shared_experts"]
    
    expected_vals = {
        "attn_absorb_wqa": {"weight_size": hidden_size*q_lora_rank}, 
        "attn_absorb_wqb": {"weight_size": q_lora_rank*num_attention_heads*(qk_nope_head_dim+qk_rope_head_dim)}, 
        "attn_absorb_wkva": {"weight_size": hidden_size*(kv_lora_rank+qk_rope_head_dim)}, 
        "attn_absorb_wkvb1": {"weight_size": num_attention_heads*qk_nope_head_dim*kv_lora_rank}, 
        "attn_absorb_absorbattn": {"kv_size": bsz*seqlen_kv*(kv_lora_rank+qk_rope_head_dim)},
        "attn_absorb_wkvb2": {"weight_size": num_attention_heads*v_head_dim*kv_lora_rank}, 
        "attn_absorb_wo": {"weight_size": num_attention_heads*v_head_dim*hidden_size}, 
        "attn_absorb_ar_tp": {},
        "moe_a2a_disp": {},
        "moe_multicast_exp": {}, 
        "moe_shared_exp": {"weight_size": moe_intermediate_size*hidden_size*3*n_shared_experts},
        "moe_exp": {"weight_size": moe_intermediate_size*hidden_size*3*n_routed_experts},
        "moe_a2a_comb": {},
        "moe_unicast": {},
        "moe_multicast_dp": {}
    }

    return expected_vals


op_info = {
    "attn_absorb_wqa": {"ax": 0, "op_type": "Linear"}, 
    "attn_absorb_wqb": {"ax": 1, "op_type": "Linear"}, 
    "attn_absorb_wkva": {"ax": 2, "op_type": "Linear"}, 
    "attn_absorb_wkvb1": {"ax": 3, "op_type": "GroupedLinear"}, 
    "attn_absorb_absorbattn": {"ax": 4, "op_type": "Attention"}, 
    "attn_absorb_wkvb2": {"ax": 5, "op_type": "GroupedLinear"}, 
    "attn_absorb_wo": {"ax": 6, "op_type": "Linear"}, 
    "attn_absorb_ar_tp": {"ax": 7, "op_type": "Comm"},
    "moe_a2a_disp": {"ax": 8, "op_type": "Comm"},
    "moe_multicast_exp": {"ax": 8, "op_type": "Comm"}, 
    "moe_shared_exp": {"ax": 9, "op_type": "Linear"},
    "moe_exp": {"ax": 9, "op_type": "Linear"},
    "moe_a2a_comb": {"ax": 10, "op_type": "Comm"},
    "moe_unicast": {"ax": 10, "op_type": "Comm"},
    "moe_multicast_dp": {"ax": 11, "op_type": "Comm"}
}

if __name__=="__main__":
    decode_iter = 0

    num_nodes = 16
    node_grid = [4, 4]
    assert num_nodes == node_grid[0] * node_grid[1]
    num_cores = 36
    core_grid = [6, 6]
    assert num_cores == core_grid[0] * core_grid[1]

    n_axes = max([op_info[k]["ax"] for k in op_info.keys()])+1
    fig, axs = plt.subplots(1, n_axes, figsize=(n_axes*10, 10))
    for ax in axs:
        plt.sca(ax)
        draw_wafer(core_grid, node_grid)

    layer = "decode5"

    stats = {op: {"bytes": {}, "byte_hop": 0, "input": 0, "weight": 0, "kv": 0, "output": 0} for op in op_info}

    for node_id in range(num_nodes):
        print("Processing traces for node {}...".format(node_id))

        for core_id in range(num_cores):
            global_core_id = core_id + node_id * num_cores

            fname = f"./traces/decode/node_{node_id}/decode{decode_iter}/core_{core_id}.csv"
            with open(fname, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")
                    op_type, tile_uid = line[0], line[1]

                    is_found = False
                    for op in op_info:
                        if f"{layer}_{op}" in tile_uid:
                            is_found = True
                            break
                    
                    if not is_found:
                        raise ValueError(f"Op {tile_uid} not found")
                    
                    if op_type == "READ":
                        src_core_id = int(line[2])
                        src = get_glob_coord(src_core_id, core_grid, node_grid)
                        dst = get_glob_coord(global_core_id, core_grid, node_grid)
                        n_bytes = int(line[3])
                        register_move(stats[op]["bytes"], src, dst, n_bytes)
                        stats[op]["byte_hop"] += n_bytes * get_num_hops(src, dst)
                        
                    elif op_type == "WRITE":
                        dst_core_id = int(line[2])
                        src = get_glob_coord(global_core_id, core_grid, node_grid)
                        dst = get_glob_coord(dst_core_id, core_grid, node_grid)
                        n_bytes = int(line[3])
                        register_move(stats[op]["bytes"], src, dst, n_bytes)
                        stats[op]["byte_hop"] += n_bytes * get_num_hops(src, dst)

                    elif op_type == "COPY":
                        src_core_id = int(line[2])
                        dst_core_id = int(line[4])
                        src = get_glob_coord(src_core_id, core_grid, node_grid)
                        dst = get_glob_coord(dst_core_id, core_grid, node_grid)
                        n_bytes = int(line[5])
                        register_move(stats[op]["bytes"], src, dst, n_bytes)
                        stats[op]["byte_hop"] += n_bytes * get_num_hops(src, dst)

                    elif op_type in ["GEMM", "REDUCE"]:
                        continue 

                    else:
                        raise NotImplementedError(f"Op type {op_type} not implemented")

                    if op_type in ["READ", "WRITE"]:
                        if op_info[op]["op_type"] in ["Linear", "GroupedLinear"]:
                            if "input" in tile_uid:
                                stats[op]["input"] += n_bytes
                            elif "weight" in tile_uid:
                                stats[op]["weight"] += n_bytes
                            elif "out" in tile_uid:
                                stats[op]["output"] += n_bytes
                            else:
                                raise NotImplementedError
                        elif op_info[op]["op_type"] == "Attention":
                            if "_q_" in tile_uid:
                                stats[op]["input"] += n_bytes
                            elif "_kv_" in tile_uid:
                                stats[op]["kv"] += n_bytes
                            elif "out" in tile_uid:
                                stats[op]["output"] += n_bytes
                            else:
                                raise NotImplementedError

    print("Data movement statistics (in MB):")
    for op in op_info:
        input_mb = stats[op]["input"] / 1024 / 1024
        weight_mb = stats[op]["weight"] / 1024 / 1024
        kv_mb = stats[op]["kv"] / 1024 / 1024
        output_mb = stats[op]["output"] / 1024 / 1024
        total_mb = input_mb + weight_mb + output_mb
        print(f"  {layer}_{op}: input={input_mb:.2f} MB, weight={weight_mb:.2f} MB, kv={kv_mb:.2f} MB, output={output_mb:.2f} MB, total={total_mb:.2f} MB")

    bsz = 128
    seqlen_kv = 2048
    expected_vals = get_expected_vals(bsz, seqlen_kv, "./configs/deepseekv3.json")

    for op in op_info:
        axis = axs[op_info[op]["ax"]]
        plt.sca(axis)

        total_bytes = 0
        for (src, dst), count in stats[op]["bytes"].items():
            total_bytes += count
            draw_arrow(src, dst, color="red", alpha=0.01*count/1024)
        
        if total_bytes == 0:
            continue

        plt.title(f"{layer}_{op}")

        expected_weight_size = expected_vals[op].get("weight_size", 0)
        expected_kv_size = expected_vals[op].get("kv_size", 0)
        plt.xlabel(
            f"""
            weight size: {expected_weight_size/1024/1024:.2f} MB\n
            KV size: {expected_kv_size/1024/1024:.2f} MB\n
            query movement: {stats[op]['input']/1024/1024:.2f} MB\n
            weight movement: {stats[op]['weight']/1024/1024:.2f} MB\n
            KV movement: {stats[op]['kv']/1024/1024:.2f} MB\n
            output movement: {stats[op]['output']/1024/1024:.2f} MB\n
            Data movement: {total_bytes/1024/1024:.2f} MB\n
            Byte-Hops: {stats[op]['byte_hop']/1024/1024:.2f} MB-Hops
            """
        )

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("wafer.png", dpi=200)


