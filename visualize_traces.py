
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

def register_move(traffic, src, dst, size):
    key = (tuple(src), tuple(dst))
    if key not in traffic:
        traffic[key] = 0
    traffic[key] += size

def get_2d_coord(core_id, core_grid):
    assert core_id < core_grid[0] * core_grid[1]

    row = core_id // core_grid[1]
    col = core_id % core_grid[1]
    return [row, col]


def get_glob_coord(core_id, core_grid, node_grid):
    num_cores_per_node = core_grid[0] * core_grid[1]
    node_id = core_id // num_cores_per_node
    local_core_id = core_id % num_cores_per_node

    node_r, node_c = get_2d_coord(node_id, node_grid)
    local_core_r, local_core_c = get_2d_coord(local_core_id, core_grid)

    glob_r = node_r * core_grid[0] + local_core_r
    glob_c = node_c * core_grid[1] + local_core_c

    return [glob_r, glob_c]

ops_to_axis = {
    "attn_absorb_wqa": 0, 
    "attn_absorb_wqb": 1, 
    "attn_absorb_wkva": 2, 
    "attn_absorb_wkvb1": 3, 
    "attn_absorb_absorbattn": 4, 
    "attn_absorb_wkvb2": 5, 
    "attn_absorb_wo": 6, 
    "attn_absorb_multicast": 7, 
    "moe_shared_exp": 8,
    "moe_exp": 8,
    "moe_unicast": 9,
    "moe_multicast": 10
}

if __name__=="__main__":
    decode_iter = 0

    num_nodes = 8
    node_grid = [4, 2]
    assert num_nodes == node_grid[0] * node_grid[1]
    num_cores = 24
    core_grid = [6, 4]
    assert num_cores == core_grid[0] * core_grid[1]

    n_axes = max([ops_to_axis[k] for k in ops_to_axis.keys()])+1
    fig, axs = plt.subplots(1, n_axes, figsize=(n_axes*10, 10))
    for ax in axs:
        plt.sca(ax)
        draw_wafer(core_grid, node_grid)

    layer = "decode5"

    traffic = {op: {} for op in ops_to_axis}

    for node_id in range(num_nodes):
        print("Generating traces for node {}...".format(node_id))

        for core_id in range(num_cores):
            global_core_id = core_id + node_id * num_cores

            fname = f"./traces/decode/node_{node_id}/decode{decode_iter}/core_{core_id}.csv"
            with open(fname, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")

                    is_found = False
                    for op in ops_to_axis:
                        if f"{layer}_{op}" in line[1]:
                            is_found = True
                            break
                    
                    if not is_found:
                        raise ValueError(f"Op {line[1]} not found")
                    
                    if line[0] == "READ":
                        src = get_glob_coord(int(line[2]), core_grid, node_grid)
                        dst = get_glob_coord(global_core_id, core_grid, node_grid)
                        n_bytes = int(line[3])
                        register_move(traffic[op], src, dst, n_bytes)
                    elif line[0] == "WRITE":
                        src = get_glob_coord(global_core_id, core_grid, node_grid)
                        dst = get_glob_coord(int(line[2]), core_grid, node_grid)
                        n_bytes = int(line[3])
                        register_move(traffic[op], src, dst, n_bytes)
                    elif line[0] == "COPY":
                        src = get_glob_coord(int(line[2]), core_grid, node_grid)
                        dst = get_glob_coord(int(line[4]), core_grid, node_grid)
                        n_bytes = int(line[5])
                        register_move(traffic[op], src, dst, n_bytes)

    for op in ops_to_axis:
        axis = axs[ops_to_axis[op]]
        plt.sca(axis)
        plt.title(f"{layer}_{op}")
        total_bytes = 0
        for (src, dst), count in traffic[op].items():
            total_bytes += count
            draw_arrow(src, dst, color="red", alpha=0.01*count/1024)
        plt.xlabel(f"Total traffic: {total_bytes/1024/1024:.2f} MB")
        
    plt.tight_layout()
    plt.savefig("wafer.png", dpi=200)


