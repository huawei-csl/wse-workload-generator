
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DrawWafer:
    def __init__(self, node_grid, core_grid) -> None:
        self.node_grid = node_grid
        self.core_grid = core_grid

        self.num_nodes = node_grid[0] * node_grid[1]
        self.num_cores_per_node = core_grid[0] * core_grid[1]
        self.num_cores = self.num_nodes * self.num_cores_per_node

        self.core_width = 0.5 
        self.core_height = 1.0
        self.bank_width = 0.5
        self.bank_height = 1.0

        self.core_coords = {}
        self.bank_coords = {}

        self.traffic = {}

        fig_h = 0.5 * node_grid[0] * core_grid[0] * self.core_height
        fig_w = 0.5 * node_grid[1] * core_grid[1] * (self.core_width + self.bank_width)
        self.figsize = (fig_w, fig_h)

        self.fig = plt.figure(figsize=self.figsize)
        self.ax  = self.fig.add_axes([0.1, 0.1, 0.75, 0.8]) # left, bottom, width, height
        # self.ax = plt.gca()

    def draw_wafer(self, with_labels=True):
        ax = self.ax

        node_grid = self.node_grid
        core_grid = self.core_grid

        # Plot the board
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_aspect('equal')
        ax.set_title("C: Core   B: Memory bank", fontsize=8)
        ax.set_xlim(-0.25, -0.25 + node_grid[1]*core_grid[1]*(self.core_width + self.bank_width))
        ax.set_ylim(-0.5, -0.5 + node_grid[0]*core_grid[0]*self.core_height)
        ax.tick_params(which="minor", size=0)  # Hide minor ticks

        for node_r in range(node_grid[0]):
            for node_c in range(node_grid[1]):
                node_id = node_r * node_grid[1] + node_c
                self.core_coords[node_id] = {}
                self.bank_coords[node_id] = {}

                offset_c = node_c*core_grid[1]
                offset_r = (node_grid[0]-1-node_r) * core_grid[0]

                # Draw the cores grid
                for core_r in range(core_grid[0]):  # Iterate over rows
                    for core_c in range(core_grid[1]):  # Iterate over columns
                        # Calculate the bottom-left corner of each rectangle
                        x = core_c + offset_c
                        y = (core_grid[0]-1)-core_r + offset_r
                        core_id = core_r * core_grid[1] + core_c

                        self.core_coords[node_id][core_id] = (x, y)

                        # Create a rectangle
                        rect = patches.Rectangle((x-0.25, y-0.5), self.core_width, self.core_height, 
                                                linewidth=1, edgecolor='black', facecolor='whitesmoke')
                        ax.add_patch(rect)

                        if with_labels:
                            ax.text(x, y, f"C{core_id}", fontsize=6, verticalalignment='center', horizontalalignment='center')

                # Draw the banks grid
                for bank_r in range(core_grid[0]):  # Iterate over rows
                    for bank_c in range(core_grid[1]):  # Iterate over columns
                        x = bank_c + offset_c + 0.5
                        y = (core_grid[0]-1)-bank_r + offset_r

                        bank_id = bank_r * core_grid[1] + bank_c

                        self.bank_coords[node_id][bank_id] = (x, y)
                        # Create a rectangle
                        rect = patches.Rectangle((x-0.25, y-0.5), self.bank_width, self.bank_height, 
                                                linewidth=1, edgecolor='black', facecolor='aliceblue')
                        ax.add_patch(rect)

                        if with_labels:
                            ax.text(x, y, f"B{bank_id}", fontsize=6, verticalalignment='center', horizontalalignment='center')

        # Draw the node grid
        for node_r in range(node_grid[0]):  # Iterate over rows
            for node_c in range(node_grid[1]):  # Iterate over columns
                # Calculate the bottom-left corner of each rectangle
                node_id = node_r * node_grid[1] + node_c
                offset_c = node_c * core_grid[1]
                offset_r = (node_grid[0]-1-node_r) * core_grid[0]

                # Create a rectangle
                rect = patches.Rectangle((offset_c-0.25, offset_r-0.5), core_grid[1], core_grid[0], 
                                        linewidth=3, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

                if with_labels:
                    ax.text(offset_c+core_grid[1]//2, offset_r+core_grid[0]//2-0.25, f"Node:{node_id}", fontsize=12, color="darkblue", verticalalignment='center', horizontalalignment='center')

    def nodeid_to_coords(self, node_id):
        row = node_id // self.node_grid[1]
        col = node_id % self.node_grid[1]
        return (row, col)
    
    def coreid_to_coords(self, core_id):
        row = core_id // self.core_grid[1]
        col = core_id % self.core_grid[1]
        return (row, col)

    def draw_arrow(self, src, dst, color="red", alpha=0.1):
        # Draw the arrow
        plt.annotate(
            "",  # No text for the annotation
            xy=(dst[0], dst[1]),  # Destination (x, y)
            xytext=(src[0], src[1]),  # Source (x, y)
            arrowprops=dict(arrowstyle="-", color=color, lw=1, alpha=min(alpha, 1.0)),  # Arrow style
        )

    def draw_traffic(self):
        max_traffic = max(list(self.traffic.values()))

        for (src, dst), size in self.traffic.items():
            if src.startswith("core"):
                _, src_node_id, src_core_id = src.split("_")
                src_node_id = int(src_node_id)
                src_core_id = int(src_core_id)
                arrow_start = self.core_coords[src_node_id][src_core_id]

            elif src.startswith("bank"):
                _, src_node_id, src_bank_id = src.split("_")
                src_node_id = int(src_node_id)
                src_bank_id = int(src_bank_id)
                arrow_start = self.bank_coords[src_node_id][src_bank_id]

            if dst.startswith("core"):
                _, dst_node_id, dst_core_id = dst.split("_")
                dst_node_id = int(dst_node_id)
                dst_core_id = int(dst_core_id)
                arrow_end = self.core_coords[dst_node_id][dst_core_id]
            
            elif dst.startswith("bank"):
                _, dst_node_id, dst_bank_id = dst.split("_")
                dst_node_id = int(dst_node_id)
                dst_bank_id = int(dst_bank_id)
                arrow_end = self.bank_coords[dst_node_id][dst_bank_id]
            
            self.draw_arrow(arrow_start, arrow_end, alpha=size/max_traffic)

        self.add_colormap(max_traffic)

    def add_colormap(self, max_value, color="red"):
        if max_value < 1024:
            title = "Bytes"
        elif max_value < 1024**2:
            title = "KBytes"
            max_value = max_value / 1024
        elif max_value < 1024**3:
            title = "MBytes"
            max_value = max_value / (1024**2)
        else:
            title = "GBytes"
            max_value = max_value / (1024**3)
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'white_to_color',
            ['white', color]
        )
        
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # required for older matplotlib versions

        # Draw colorbar only
        bar_width = 0.01
        cax = self.fig.add_axes([0.88, 0.2, bar_width, 0.6]) # Colorbar axes: [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cax, shrink=0.01)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_title(
            "       " + title,
            pad=8,      # space above the colorbar
            fontsize=8
        )

    def register_traffic(self, src, dst, size):
        key = (src, dst)
        if key not in self.traffic:
            self.traffic[key] = 0
        self.traffic[key] += size

    def register_bank_to_core(self, src_node_id, src_bank_id, dst_node_id, dst_core_id, size):
        self.register_traffic(f"bank_{src_node_id}_{src_bank_id}", f"core_{dst_node_id}_{dst_core_id}", size)

    def register_core_to_bank(self, src_node_id, src_core_id, dst_node_id, dst_bank_id, size):
        self.register_traffic(f"core_{src_node_id}_{src_core_id}", f"bank_{dst_node_id}_{dst_bank_id}", size)

    def register_bank_to_bank(self, src_node_id, src_bank_id, dst_node_id, dst_bank_id, size):
        self.register_traffic(f"bank_{src_node_id}_{src_bank_id}", f"bank_{dst_node_id}_{dst_bank_id}", size)

    def save(self, fname):
        plt.savefig(fname, dpi=300)

if __name__=="__main__":
    node_grid = (4, 1)
    core_grid = (6, 6)

    wafer = DrawWafer(node_grid, core_grid)
    
    wafer.draw_wafer()
    wafer.register_bank_to_core(1, 3, 2, 8, 256)
    wafer.register_core_to_bank(0, 30, 0, 15, 256)
    wafer.register_bank_to_bank(2, 5, 3, 10, 128)
    wafer.draw_traffic()

    wafer.save("visualize/images/wafer.png")