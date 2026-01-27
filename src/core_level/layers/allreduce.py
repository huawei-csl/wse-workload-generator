
import logging

from typing import List

from src.core_level.common.tensor import Tensor
from src.core_level.common.isa import InstructionSet
from src.core_level.common.stats import Stats
from src.core_level.layers.barrier import Barrier
from src.core_level.layers.reduce import TileReduceOp

from src.node_level.common.utils import byte_to_str, dtype_to_byte

class TileAllreduceStage1Op:
    def __init__(self, id, send0_tile, next_tile, comm_group) -> None:
        self.id = id
        self.send0_tile = send0_tile
        self.next_tile = next_tile
        self.mapped_core = None
        self.comm_group = comm_group
        self.stats = Stats()
        logging.debug("TileAllreduceStage1Op {} is created.".format(self.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileAllreduceStage1Op {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileAllreduceStage1Op {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))
    
    def get_traces(self) -> List[str]:
        traces = []

        send0_mem_sizes = self.send0_tile.get_physical_address()
        next_mem_sizes = self.next_tile.get_physical_address()

        assert len(send0_mem_sizes) == len(next_mem_sizes), "Mismatched number of memory banks between send0 and next tile in TileAllreduceStage1Op {}.".format(self.id)
        for i in range(len(send0_mem_sizes)):
            send0_bank = list(send0_mem_sizes.keys())[i]
            send0_size = send0_mem_sizes[send0_bank]
            next_bank = list(next_mem_sizes.keys())[i]
            next_size = next_mem_sizes[next_bank]

            assert send0_size == next_size, "Mismatched send0 and next tile sizes in TileAllreduceStage1Op {}.".format(self.id)
            
            traces.append(InstructionSet.COPY(send0_bank.bank_id, next_bank.bank_id, send0_size, self.id))
            self.stats.add_copy(send0_size)

        return traces


class TileAllreduceStage2Op:
    def __init__(self, id, send_tile, next_tile, comm_group) -> None:
        self.id = id
        self.send_tile = send_tile
        self.next_tile = next_tile
        self.mapped_core = None
        self.comm_group = comm_group
        self.stats = Stats()
        logging.debug("TileAllreduceStage2Op {} is created.".format(self.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileAllreduceStage2Op {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileAllreduceStage2Op {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))
    
    def get_traces(self) -> List[str]:
        traces = []

        send_mem_sizes = self.send_tile.get_physical_address()
        next_mem_sizes = self.next_tile.get_physical_address()

        assert len(send_mem_sizes) == len(next_mem_sizes), "Mismatched number of memory banks between send0 and next tile in TileAllreduceStage1Op {}.".format(self.id)

        for i in range(len(send_mem_sizes)):
            send_bank = list(send_mem_sizes.keys())[i]
            send_size = send_mem_sizes[send_bank]
            next_bank = list(next_mem_sizes.keys())[i]
            next_size = next_mem_sizes[next_bank]

            assert send_size == next_size, "Mismatched send0 and next tile sizes in TileAllreduceStage1Op {}.".format(self.id)
            
            traces.append(InstructionSet.COPY(send_bank.bank_id, next_bank.bank_id, send_size, self.id))
            self.stats.add_copy(send_size)

        return traces
        
class AllreduceLayer:
    def __init__(self, uid, node_id, comm_group, graph, dims, wafer, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.comm_group = comm_group
        self.dims = dims
        self.wafer = wafer

        # assert len(dims) == 2, "AllreduceLayer currently only supports 2D tensors."

        self.graph_op = graph.get_op(node_id, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=dims,
            prec=prec,
        )
        assert self.input_tensor.tile_size is not None, "Input tensor {} of Allreduce operation {} on node {} does not have tile size.".format(self.input_tensor.uid, uid, self.node_id)
        self.tile_size = list(self.input_tensor.tile_size)

        self.output_tensor = self.input_tensor.clone(self.graph_op["outputs"][0])
        
        next_node = comm_group[(comm_group.index(node_id) + 1) % len(comm_group)]
        self.next_node = next_node
        next_op = graph.get_op(next_node, uid)

        self.next_tensor = Tensor(
            uid=next_op["inputs"][0],
            dims=dims,
            prec=prec,
        )
        
        # assert dims[-1] % len(comm_group) == 0, "Vector dimension must be divisible by the number of nodes in the communication group."
        # self.tile_size = self.input_tensor.tile_size

        self.send_tiles = {}
        self.recv_tiles = {}
        self.next_tiles = {}
        self.stage1_ops = {}
        self.reduce_ops = {}
        self.stage2_ops = {}
        
        self.stats = Stats()

        self.create_tiles()
        self.create_ops()

        self.map_ops()

    def create_tiles(self):
        self.input_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=self.tile_size, addr_offset=0)
        self.output_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=self.tile_size, addr_offset=0)
        self.next_tensor.map_to_memory(self.wafer.banks[self.next_node], tile_size=self.tile_size, addr_offset=0)

        chunk_size = self.dims[-1] // len(self.comm_group)
        for d0, pD0 in enumerate(range(0, self.dims[-1], chunk_size)):
            tiled_B = min(chunk_size, self.dims[-1] - pD0)

            slice_indices = [(0, d) for d in self.dims[0:-1]] + [(pD0, pD0 + tiled_B)]


            self.send_tiles[d0] = self.input_tensor.slice(slice_indices)
            self.recv_tiles[d0] = self.output_tensor.slice(slice_indices)
            self.next_tiles[d0] = self.next_tensor.slice(slice_indices)
        
    def create_ops(self):
        '''
        Ring implementation of Allreduce, implemented as a two-stage process: reduce followed by gather.
        In the reduce stage, each node sends its data chunk to the next node in the communication group
        and receives a data chunk from the previous node, performing element-wise addition on the received.
        For example, in a 4-node group (0, 1, 2, 3):
        Round 0:
            Node 0 sends chunk 0 to Node 1, receives chunk 3 from Node 3
            Node 1 sends chunk 1 to Node 2, receives chunk 0 from Node 0
            Node 2 sends chunk 2 to Node 3, receives chunk 1 from Node 1
            Node 3 sends chunk 3 to Node 0, receives chunk 2 from Node 2
        Round 1:
            Node 0 sends chunk 3 to Node 1, receives chunk 2 from Node 3
            Node 1 sends chunk 0 to Node 2, receives chunk 3 from Node 0
            Node 2 sends chunk 1 to Node 3, receives chunk 0 from Node 1
            Node 3 sends chunk 2 to Node 0, receives chunk 1 from Node 2
        Round 2:
            Node 0 sends chunk 2 to Node 1, receives chunk 1 from Node 3
            Node 1 sends chunk 3 to Node 2, receives chunk 2 from Node 0
            Node 2 sends chunk 0 to Node 3, receives chunk 3 from Node 1
            Node 3 sends chunk 1 to Node 0, receives chunk 0 from Node 2
        At the end of Round 2:
            Node 0 has the sum of chunk 1
            Node 1 has the sum of chunk 2
            Node 2 has the sum of chunk 3
            Node 3 has the sum of chunk 0
        Now, in the gather stage, each node sends the reduced chunk to the next node
        and receives the reduced chunk from the previous node, storing it in the appropriate position.
        Round 0:
            Node 0 sends chunk 1 to Node 1, receives chunk 0 from Node 3
            Node 1 sends chunk 2 to Node 2, receives chunk 1 from Node 0
            Node 2 sends chunk 3 to Node 3, receives chunk 2 from Node 1
            Node 3 sends chunk 0 to Node 0, receives chunk 3 from Node 2
        Round 1:
            Node 0 sends chunk 0 to Node 1, receives chunk 3 from Node 3
            Node 1 sends chunk 1 to Node 2, receives chunk 0 from Node 0
            Node 2 sends chunk 2 to Node 3, receives chunk 1 from Node 1
            Node 3 sends chunk 3 to Node 0, receives chunk 2 from Node 2
        Round 2:
            Node 0 sends chunk 3 to Node 1, receives chunk 2 from Node 3
            Node 1 sends chunk 0 to Node 2, receives chunk 3 from Node 0
            Node 2 sends chunk 1 to Node 3, receives chunk 0 from Node 1
            Node 3 sends chunk 2 to Node 0, receives chunk 1 from Node 2
        At the end of Round 2, each node has the complete reduced vector.
        '''

        num_rounds = len(self.comm_group) - 1
        # Reduce stage
        for i in range(num_rounds):
            send_chunk_id = (self.node_id - i) % len(self.comm_group)
            recv_chunk_id = (self.node_id - i - 1) % len(self.comm_group)
            logging.debug("Reduce round {}: chunk{} node{} -> node{}".format(i, send_chunk_id, self.node_id, self.next_node))

            # Op to copy a chunk to the next node
            self.stage1_ops[i] = TileAllreduceStage1Op("{}_stage1_{}".format(self.uid, i), self.send_tiles[send_chunk_id], self.next_tiles[send_chunk_id], self.comm_group)
            
            # Op to perform element-wise addition on the received chunk
            self.reduce_ops[i] = TileReduceOp("{}_reduce_{}".format(self.uid, i), [self.send_tiles[recv_chunk_id], self.recv_tiles[recv_chunk_id]], self.send_tiles[recv_chunk_id])  # Placeholder for potential future use
        
        # Gather stage
        for i in range(num_rounds):
            send_chunk_id = (self.node_id - i + 1) % len(self.comm_group)
            logging.debug("Gather round {}: chunk{} node{} -> node{}".format(i, send_chunk_id, self.node_id, self.next_node))
            
            # Op to copy the final chunk to the next node in the gather stage
            self.stage2_ops[i] = TileAllreduceStage2Op("{}_stage2_{}".format(self.uid, i), self.send_tiles[send_chunk_id], self.next_tiles[send_chunk_id], self.comm_group)

    def map_ops(self):
        dedicated_core = self.wafer.get_core(self.node_id, 0)
        num_rounds = len(self.comm_group) - 1
        for i in range(num_rounds):
            mem_sizes = self.stage1_ops[i].send0_tile.get_physical_address()
            # assert len(mem_sizes) == 1, "AllreduceLayer currently only supports mapping 2D tiles to a single core."
            local_bank_id = list(mem_sizes.keys())[0].local_id
            # self.tile_ops[i].map_to_core(self.wafer.get_core(self.node_id, local_bank_id))
            self.stage1_ops[i].map_to_core(dedicated_core)
            self.stats.merge(self.stage1_ops[i].stats)

            cores_to_sync = [self.wafer.get_core(node_id, 0) for node_id in self.comm_group]
            barrier = Barrier(f"{self.uid}_barrier_stage1_{i}", cores_to_sync)
            barrier.map_to_core(dedicated_core)

            self.reduce_ops[i].map_to_core(dedicated_core)
            self.stats.merge(self.reduce_ops[i].stats)

            cores_to_sync = [self.wafer.get_core(node_id, 0) for node_id in self.comm_group]
            barrier = Barrier(f"{self.uid}_barrier_reduce_{i}", cores_to_sync)
            barrier.map_to_core(dedicated_core)
            
        for i in range(num_rounds):
            mem_sizes = self.stage2_ops[i].send_tile.get_physical_address()
            # assert len(mem_sizes) == 1, "AllreduceLayer currently only supports mapping 2D tiles to a single core."
            local_bank_id = list(mem_sizes.keys())[0].local_id
            # self.tile_ops[num_rounds+i].map_to_core(self.wafer.get_core(self.node_id, local_bank_id))
            self.stage2_ops[i].map_to_core(dedicated_core)
            self.stats.merge(self.stage2_ops[i].stats)

            cores_to_sync = [self.wafer.get_core(node_id, 0) for node_id in self.comm_group]
            barrier = Barrier(f"{self.uid}_barrier_gather_{i}", cores_to_sync)
            barrier.map_to_core(dedicated_core)
            
    def calc_expected(self):
        vector_size = eval("*".join(map(str, self.dims))) * dtype_to_byte(self.input_tensor.prec)
        num_rounds = len(self.comm_group) - 1

        chunk_size = vector_size / len(self.comm_group)
        total_copy = 2 * num_rounds * chunk_size # total data copied per node for both reduce and gather stages
        expected = {"copy": total_copy}
        
        expected["reads"] = chunk_size * 2 * num_rounds # in each round, each node reads two chunks, add them, and writes one back
        expected["writes"] = chunk_size * num_rounds 
        expected["vector_flops"] = eval("*".join(map(str, self.dims))) / len(self.comm_group) * num_rounds 

        return expected

    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, expected=expected, dims=self.dims, tile_size=self.tile_size)



if __name__ == "__main__":
    from src.core_level.common.wafer import Wafer
    from src.core_level.common.tensor import reset_tensor_registry
    from src.core_level.common.graph import Graph

    reset_tensor_registry()

    node_grid = (2, 2)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    ops = {}
    for node_id in range(wafer.num_nodes):
        ops[node_id] = {}
        op_id = f"allreduce_0"
        ops[node_id][op_id] = {
            "type": "Allreduce",
            "inputs": [f"{node_id}:input_tensor"],
            "outputs": [f"{node_id}:output_tensor"]
        }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    dims = [4, 4, 4]
    comm_group = [0, 1]
    tile_size = [2, 2, 2]

    input_tensor = Tensor(
        uid="0:input_tensor",
        dims=dims,
        prec="fp16",
    )
    input_tensor.map_to_memory(wafer.banks[0], tile_size=tile_size, addr_offset=0)

    for node_id in comm_group:
        layer = AllreduceLayer(f"allreduce_0", node_id, comm_group, graph, dims, wafer, "fp16")
        expected = layer.calc_expected()
        assert expected["copy"] == layer.stats.get_copy(), "AllreduceLayer stats do not match expected values."
        assert expected["reads"] == layer.stats.get_reads(), "AllreduceLayer stats do not match expected values."
        assert expected["writes"] == layer.stats.get_writes(), "AllreduceLayer stats do not match expected values."
        assert expected["vector_flops"] == layer.stats.get_vector(), "AllreduceLayer stats do not match expected values."

        layer.log_stats()

    traces = wafer.get_traces()
    for node_id in traces:
        print("\n=== Node {} Traces ===".format(node_id))
        for core_id in traces[node_id]:
            print("-- Core {} --".format(core_id))
            for inst in traces[node_id][core_id]:
                print(inst)