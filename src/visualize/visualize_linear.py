from src.core_level.layers.linear import LinearLayer
from src.core_level.common.wafer import Wafer
from src.core_level.common.graph import Graph

from src.visualize.draw_wafer import DrawWafer

if __name__=="__main__":
    node_grid = (2, 2)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    draw = DrawWafer(wafer)

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"{node_id}:linear_0"
        ops[node_id][op_id] = {
            "type": "Linear",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    dims = (128, 128, 128)
    tile_size = (16, 16, 16)

    for node_id in range(wafer.num_nodes):
        layer = LinearLayer(f"{node_id}:linear_0", node_id, graph, dims, tile_size, wafer, prec="fp16")

    traces = wafer.get_traces()
    draw.draw_traces(traces, out_path="output/visuals/linear_0_traces.png")