import logging 
import pytest

from core_level.layers.linear import LinearLayer
from core_level.common.wafer import Wafer
from core_level.common.graph import Graph
from core_level.common.isa import InstructionSet
from core_level.common.tensor import reset_tensor_registry

@pytest.mark.parametrize(
    "node_grid,dims,tile_size,expected", 
    [
        # single node, no tiling test case
        (
            [1, 1], 
            [16, 16, 16], 
            [16, 16, 16], 
            {
                "reads": (16*16 + 16*16) * 2,  # input + weight, in fp16 (2 bytes)
                "writes": (16*16) * 2,         # output, in fp16 (2 bytes)
                "flops": 2*16*16*16
            }
        ),
        # multi-node, with tiling test case, no split-K
        (
            [2, 2], 
            [32, 16, 32], 
            [16, 16, 16], 
            {
                "reads": 4 * ( 4*(16*16 + 16*16) * 2 ),    # input + weight, in fp16 (2 bytes)
                "writes": 4 * (32*32) * 2,                 # output, in fp16 (2 bytes)
                "flops": 4 * (2*32*16*32)
            }
        ),
        # multi-node, with tiling test case, with split-K
        (
            [4, 4], 
            [32, 32, 32], 
            [16, 16, 16], 
            {
                "reads": 16 * ( 8 * (16*16 + 16*16) * 2 + 8 * (16*16) * 2 ),    # input + weight + partial sums, in fp16 (2 bytes)
                "writes": 16 * (8 * (16*16) * 2 + (32*32) * 2),             # partial sums + output, in fp16 (2 bytes)
                "flops": 16 * (8 * (16*16) + 2 * (32*32*32))               # gemm flops + add flops
            }
        ),
    ]
)
def test_linear(node_grid, dims, tile_size, expected):
    reset_tensor_registry()

    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

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

    for node_id in range(wafer.num_nodes):
        layer = LinearLayer(f"{node_id}:linear_0", node_id, graph, dims, tile_size, wafer, prec="fp16")

    traces = wafer.get_traces()

    total_reads = 0
    total_writes = 0
    total_flops = 0
    for node_id in range(wafer.num_nodes):
        for core_id in range(wafer.num_cores_per_node):
            for trace in traces[node_id][core_id]:
                parsed_instr = InstructionSet.parse(trace)
                # print(f"{node_id}:{core_id}:", parsed_instr)
                if parsed_instr[0] == "READ":
                    # parsed core/bank ids are global ids, convert to local
                    src_node = parsed_instr[1] // wafer.num_cores_per_node
                    src_bank = parsed_instr[1] % wafer.num_cores_per_node
                    size = parsed_instr[2]
                    total_reads += size
                    logging.debug(f"READ from {src_node}:{src_bank} size {size}")
                    assert src_node == node_id, "Source node must be the current node"

                elif parsed_instr[0] == "WRITE":
                    dst_node = parsed_instr[1] // wafer.num_cores_per_node
                    dst_bank = parsed_instr[1] % wafer.num_cores_per_node
                    size = parsed_instr[2] 
                    total_writes += size
                    logging.debug(f"WRITE to {dst_node}:{dst_bank} size {size}")
                    assert dst_node == node_id, "Destination node must be the current node"

                elif parsed_instr[0] == "GEMM":
                    M, K, N = parsed_instr[1], parsed_instr[2], parsed_instr[3]
                    logging.debug(f"GEMM M:{M} K:{K} N:{N}")
                    total_flops += 2*M*K*N

                elif parsed_instr[0] == "ADD":
                    dims = parsed_instr[1]
                    logging.debug(f"ADD dims:{dims}")
                    total_flops += eval("*".join(map(str, dims)) )
                else:
                    raise ValueError(f"Unknown instruction {parsed_instr[0]}")

    assert total_reads == expected["reads"], f"Expected {expected['reads']} reads, got {total_reads}."
    assert total_writes == expected["writes"], f"Expected {expected['writes']} writes, got {total_writes}."
    assert total_flops == expected["flops"], f"Expected {expected['flops']} flops, got {total_flops}."

if __name__=="__main__":
    node_grid = [2, 2]
    dims = [16, 7168, 576]
    tile_size = [16, 1024, 64]
    expected = {}

    n_tile_ops = (dims[0] // tile_size[0]) * (dims[1] // tile_size[1]) * (dims[2] // tile_size[2])
    expected["reads"] = node_grid[0] * node_grid[1] * n_tile_ops * (tile_size[0]*tile_size[1] + tile_size[1]*tile_size[2]) * 2  # input + weight, in fp16 (2 bytes)

    if dims[1] != tile_size[1]:
        # if K dimension is tiled, need to read partial sums
        expected["reads"] += node_grid[0] * node_grid[1] * (dims[0] // tile_size[0]) * (dims[2] // tile_size[2]) * (dims[1] // tile_size[1]) * (tile_size[0] * tile_size[2]) * 2  # partial sums, in fp16 (2 bytes)
    
    expected["writes"] = 0
    if dims[1] != tile_size[1]:
        # if K dimension is tiled, need to write partial sums
        expected["writes"] = node_grid[0] * node_grid[1] * n_tile_ops * (tile_size[0] * tile_size[2]) * 2  # partial sums, in fp16 (2 bytes)

    expected["writes"] += node_grid[0] * node_grid[1] * (dims[0] * dims[2]) * 2  # output, in fp16 (2 bytes)

    expected["flops"] = node_grid[0] * node_grid[1] * (2*dims[0]*dims[1]*dims[2])

    if dims[1] != tile_size[1]:
        # if K dimension is tiled, need to add partial sums
        expected["flops"] += node_grid[0] * node_grid[1] * (dims[1] // tile_size[1]) * (dims[0]*dims[2])

    print(f"Input size: {node_grid[0]*node_grid[1]*dims[0]*dims[1]*2} B, Weight size: {node_grid[0]*node_grid[1]*dims[1]*dims[2]*2} B, Total reads: {expected['reads']} B")
    print(f"Output size: {node_grid[0]*node_grid[1]*dims[0]*dims[2]*2} B, Total writes: {expected['writes']} B")

    test_linear(
        node_grid,
        dims, 
        tile_size,
        expected
    )



