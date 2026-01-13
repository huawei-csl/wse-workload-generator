
import logging

from typing import List

from core_level.common.tensor import Tensor
from core_level.common.isa import InstructionSet
from core_level.common.stats import Stats
from utils import byte_to_str

class TileAllreduceStage1Op:
    def __init__(self, id, send0_tile, send1_tile, recv_tile, next_tile) -> None:
        self.id = id
        self.send0_tile = send0_tile
        self.send1_tile = send1_tile
        self.recv_tile = recv_tile
        self.next_tile = next_tile
        self.mapped_core = None
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
            self.stats.add_reads(send0_size)
            self.stats.add_writes(send0_size)
            # stats["reads"] += send0_size
            # stats["writes"] += send0_size

        # Read send1_tile from memory
        mem_sizes = self.send1_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
            self.stats.add_reads(size)
            # stats["reads"] += size

        # Read send1_tile from memory
        mem_sizes = self.recv_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.READ(bank.bank_id, size, self.id))
            self.stats.add_reads(size)
            # stats["reads"] += size

        # Element-wise addition: send1_tile += recv_tile
        traces.append(InstructionSet.ADD(self.send1_tile.dims, self.id))
        self.stats.add_vector(self.mapped_core.core_id, eval("*".join(map(str, self.send1_tile.dims))))
        # stats["flops"] += eval("*".join(map(str, self.send1_tile.dims)) )

        # Write output tile back to memory
        mem_sizes = self.send1_tile.get_physical_address()
        for bank, size in mem_sizes.items():
            traces.append(InstructionSet.WRITE(bank.bank_id, size, self.id))
            self.stats.add_writes(size)
            # stats["writes"] += size

        return traces


class TileAllreduceStage2Op:
    def __init__(self, id, send_tile, next_tile) -> None:
        self.id = id
        self.send_tile = send_tile
        self.next_tile = next_tile
        self.mapped_core = None
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
            self.stats.add_reads(send_size)
            self.stats.add_writes(send_size)
            # stats["reads"] += send_size
            # stats["writes"] += send_size

        return traces
        
class AllreduceLayer:
    def __init__(self, uid, node_id, comm_group, graph, dims, wafer, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.comm_group = comm_group
        self.dims = dims
        self.wafer = wafer

        assert len(dims) == 2, "AllreduceLayer currently only supports 2D tensors."

        self.graph_op = graph.get_op(node_id, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=dims,
            prec=prec,
        )
        self.output_tensor = self.input_tensor.clone(self.graph_op["outputs"][0])
        
        next_node = comm_group[(comm_group.index(node_id) + 1) % len(comm_group)]
        self.next_node = next_node
        next_op = graph.get_op(next_node, uid)
        self.next_tensor = self.input_tensor.clone(next_op["outputs"][0])
        
        assert dims[-1] % len(comm_group) == 0, "Vector dimension must be divisible by the number of nodes in the communication group."
        self.tile_size = [dims[0], dims[-1] // len(comm_group)]

        self.send_tiles = {}
        self.recv_tiles = {}
        self.next_tiles = {}
        self.tile_ops = {}

        self.stats = Stats()

        self.create_tiles()
        self.create_ops()

        self.map_ops()

    def create_tiles(self):
        self.input_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=self.tile_size, addr_offset=0)
        self.output_tensor.map_to_memory(self.wafer.banks[self.node_id], tile_size=self.tile_size, addr_offset=0)
        self.next_tensor.map_to_memory(self.wafer.banks[self.next_node], tile_size=self.tile_size, addr_offset=0)

        chunk_size = self.tile_size[-1]
        for d, pD in enumerate(range(0, self.dims[-1], chunk_size)):
            tiled_B = min(chunk_size, self.dims[-1] - pD)
            self.send_tiles[d] = self.input_tensor.slice([(0, self.dims[0]), (pD, pD + tiled_B)])
            self.recv_tiles[d] = self.output_tensor.slice([(0, self.dims[0]), (pD, pD + tiled_B)])
            self.next_tiles[d] = self.next_tensor.slice([(0, self.dims[0]), (pD, pD + tiled_B)])
        
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
        for i in range(num_rounds):
            send_chunk_id = (self.node_id - i) % len(self.comm_group)
            recv_chunk_id = (self.node_id - i - 1) % len(self.comm_group)
            logging.debug("Reduce round {}: chunk{} node{} -> node{}".format(i, send_chunk_id, self.node_id, self.next_node))
            self.tile_ops[i] = TileAllreduceStage1Op("{}_reduce_{}".format(self.uid, i), self.send_tiles[send_chunk_id], self.send_tiles[recv_chunk_id], self.recv_tiles[recv_chunk_id], self.next_tiles[send_chunk_id])
        
        for i in range(num_rounds):
            send_chunk_id = (self.node_id - i + 1) % len(self.comm_group)
            logging.debug("Gather round {}: chunk{} node{} -> node{}".format(i, send_chunk_id, self.node_id, self.next_node))
            self.tile_ops[num_rounds+i] = TileAllreduceStage2Op("{}_gather_{}".format(self.uid, i), self.send_tiles[send_chunk_id], self.next_tiles[send_chunk_id])

    def map_ops(self):
        num_rounds = len(self.comm_group) - 1
        for i in range(num_rounds):
            mem_sizes = self.tile_ops[i].send0_tile.get_physical_address()
            # assert len(mem_sizes) == 1, "AllreduceLayer currently only supports mapping 2D tiles to a single core."
            local_bank_id = list(mem_sizes.keys())[0].local_id
            self.tile_ops[i].map_to_core(self.wafer.get_core(self.node_id, local_bank_id))
            self.stats.merge(self.tile_ops[i].stats)

        for i in range(num_rounds):
            mem_sizes = self.tile_ops[num_rounds+i].send_tile.get_physical_address()
            # assert len(mem_sizes) == 1, "AllreduceLayer currently only supports mapping 2D tiles to a single core."
            local_bank_id = list(mem_sizes.keys())[0].local_id
            self.tile_ops[num_rounds+i].map_to_core(self.wafer.get_core(self.node_id, local_bank_id))
            self.stats.merge(self.tile_ops[num_rounds+i].stats)

    def print_stats(self):
        self.stats.print_stats(self.uid)