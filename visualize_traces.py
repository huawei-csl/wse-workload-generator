
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

def draw_arrow(src, dst, color="red"):
    # Draw the arrow
    plt.annotate(
        "",  # No text for the annotation
        xy=(dst[1], dst[0]),  # Destination (x, y)
        xytext=(src[1], src[0]),  # Source (x, y)
        arrowprops=dict(arrowstyle="->", color=color, lw=1, alpha=0.1),  # Arrow style
    )

def get_core_coord(core_id, core_grid, offset=[0,0]):
    row = core_id // core_grid[1] + offset[0]
    col = core_id % core_grid[1] + offset[1]
    return [row, col]

ops = ["attn_absorb_wqa", "attn_absorb_wqb", "attn_absorb_wkva", "attn_absorb_wkvb1", "attn_absorb_absorbattn", "attn_absorb_wkvb2", "attn_absorb_wo", "moe"]

if __name__=="__main__":
    decode_iter = 0

    num_nodes = 16
    node_grid = [4, 4]
    assert num_nodes == node_grid[0] * node_grid[1]
    num_cores = 24
    core_grid = [6, 4]
    assert num_cores == core_grid[0] * core_grid[1]

    fig, axs = plt.subplots(1, len(ops), figsize=(len(ops)*10, 10))
    for ax in axs:
        plt.sca(ax)
        draw_wafer(core_grid, node_grid)

    layer = "decode5"

    for node_id in range(num_nodes):
        node_coord = get_core_coord(node_id, node_grid)
        core_offset = [node_coord[0] * core_grid[0], node_coord[1] * core_grid[1]]

        for core_id in range(num_cores):
            fname = f"./traces/decode/node_{node_id}/decode{decode_iter}/core_{core_id}.csv"
            with open(fname, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")

                    is_found = False
                    for op in ops:
                        if f"{layer}_{op}" in line[1]:
                            is_found = True
                            break
                    
                    if not is_found:
                        raise ValueError(f"Op {line[1]} not found")
                        # continue

                    op_ind = ops.index(op)
                    plt.sca(axs[op_ind])
                    plt.title(f"{layer}_{op}")

                    if line[0] == "READ":
                        src = get_core_coord(int(line[2]), core_grid, core_offset)
                        dst = get_core_coord(core_id, core_grid, core_offset)
                        draw_arrow(src, dst, color="red")
                    elif line[0] == "WRITE":
                        src = get_core_coord(core_id, core_grid, core_offset)
                        dst = get_core_coord(int(line[2]), core_grid, core_offset) 
                        draw_arrow(src, dst, color="purple")

    plt.tight_layout()
    plt.savefig("wafer.png", dpi=300)


