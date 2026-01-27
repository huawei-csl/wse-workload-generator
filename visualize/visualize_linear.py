import os 
import shutil
import logging 
import matplotlib.pyplot as plt

from src.core_level.layers.linear import LinearLayer
from src.core_level.common.wafer import Wafer
from src.core_level.common.graph import Graph
from src.core_level.common.isa import InstructionSet

from visualize.draw_wafer import DrawWafer

if __name__=="__main__":
    node_grid = (2, 2)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    draw = DrawWafer(node_grid, core_grid)

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

    for node_id in range(wafer.num_nodes):
        for core_id in range(wafer.num_cores_per_node):
            for trace in traces[node_id][core_id]:
                parsed_instr = InstructionSet.parse(trace)
                print(f"{core_id}:", parsed_instr)
                if parsed_instr[0] == "READ":
                    # parsed core/bank ids are global ids, convert to local
                    src_node = parsed_instr[1] // wafer.num_cores_per_node
                    src_bank = parsed_instr[1] % wafer.num_cores_per_node
                    size = parsed_instr[2]
                    draw.register_bank_to_core(src_node, src_bank, node_id, core_id, size)
                    logging.debug(f"READ from {src_node}:{src_bank} size {size}")
                elif parsed_instr[0] == "WRITE":
                    dst_node = parsed_instr[1] // wafer.num_cores_per_node
                    dst_bank = parsed_instr[1] % wafer.num_cores_per_node
                    size = parsed_instr[2] 
                    draw.register_core_to_bank(node_id, core_id, dst_node, dst_bank, size)
                    logging.debug(f"WRITE to {dst_node}:{dst_bank} size {size}")
                elif parsed_instr[0] == "COPY":
                    src_node = parsed_instr[1] // wafer.num_banks_per_node
                    assert src_node == node_id, "Source node must be the current node"
                    src_bank = parsed_instr[1] % wafer.num_banks_per_node
                    dst_node = parsed_instr[2] // wafer.num_banks_per_node
                    dst_bank = parsed_instr[2] % wafer.num_banks_per_node
                    size = parsed_instr[3]
                    draw.register_bank_to_bank(src_node, src_bank, dst_node, dst_bank, size)
                    logging.debug(f"COPY from {src_node}:{src_bank} to {dst_node}:{dst_bank} size {size}")
                else:
                    pass

    draw.draw_wafer()
    draw.draw_traffic()
    draw.save("visualize/images/linear_0.png")
