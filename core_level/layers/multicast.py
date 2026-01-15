
import logging

from typing import List
from utils import dtype_to_byte

from core_level.common.stats import Stats
from core_level.common.isa import InstructionSet
from core_level.common.tensor import Tensor

class TileMulticastOp:
    def __init__(self, id, input_tile, out_tiles) -> None:
        self.id = id
        self.input_tile = input_tile
        self.out_tiles = out_tiles
        self.mapped_core = None
        self.stats = Stats()
        logging.debug("TileMulticastOp {} is created with input tile {}, out tiles {}.".format(self.id, self.input_tile.id, [t.id for t in self.out_tiles]))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileMulticastOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileMulticastOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        traces = []

        send_mem_sizes = self.input_tile.get_physical_address()
        for i in range(len(send_mem_sizes)):
            send_bank = list(send_mem_sizes.keys())[i]
            send_size = send_mem_sizes[send_bank]

            recv_banks = []
            for d in range(len(self.out_tiles)):
                recv_mem_sizes = self.out_tiles[d].get_physical_address()
                assert len(send_mem_sizes) == len(recv_mem_sizes), "Mismatched number of memory banks between send0 and next tile in TileAllreduceStage1Op {}.".format(self.id)

                recv_bank = list(recv_mem_sizes.keys())[i]
                recv_size = recv_mem_sizes[recv_bank]

                assert send_size == recv_size, "Mismatched send0 and next tile sizes in TileAllreduceStage1Op {}.".format(self.id)
                
                recv_banks.append(recv_bank.bank_id)

            traces.append(InstructionSet.MULTICAST(send_bank.bank_id, recv_banks, send_size, self.id))
            self.stats.add_multicast(send_size)

        return traces
    


class MulticastLayer:
    ''' Multicast Layer.
    Args:
        uid: unique identifier for the layer
        src: source node index
        dsts: list of destination node indices
        graph: compute graph object
        dims: dimensions of the vector to be multicasted
        cores: list of Core objects available for mapping operations
        banks: list of MemoryBank objects available for mapping tiles
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, src, dsts, graph, dims, wafer, prec) -> None:
        self.uid = uid
        self.src = src
        self.dsts = dsts 
        self.dims = dims 
        self.wafer = wafer 
        self.prec = prec

        self.graph_op = graph.get_op(src, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=dims,
            prec=self.prec
        )
        self.tile_size = self.input_tensor.tile_size

        self.output_tensors = []
        for d, dst in enumerate(dsts):
            output_tensor = Tensor(
                uid=self.graph_op["outputs"][d],
                dims=dims,
                prec=self.prec,
            )
            self.output_tensors.append(output_tensor)

        self.in_tiles = {}
        self.out_tiles = {}
        self.tile_ops = {}

        self.stats = Stats()

        self.create_tiles()
        self.create_ops()

        self.map_ops()

    def create_tiles(self):
        self.input_tensor.map_to_memory(self.wafer.banks[self.src], tile_size=self.tile_size, addr_offset=0)
        for d, dst in enumerate(self.dsts):
            self.output_tensors[d].map_to_memory(self.wafer.banks[dst], tile_size=self.tile_size, addr_offset=0)

        # Tile from last dimension
        for d0, pD0 in enumerate(range(0, self.dims[-1], self.tile_size[-1])):
            tiled_B = min(self.tile_size[-1], self.dims[-1] - pD0)

            slice_indices = [(0, d) for d in self.dims[0:-1]] + [(pD0, pD0 + tiled_B)]

            self.in_tiles[d0] = self.input_tensor.slice(slice_indices)
            self.out_tiles[d0] = []
            for d, dst_node in enumerate(self.dsts):
                self.out_tiles[d0].append(self.output_tensors[d].slice(slice_indices))

    def create_ops(self):
        for b in self.in_tiles:
            self.tile_ops[b] = TileMulticastOp("{}_multicast_{}".format(self.uid, b), self.in_tiles[b], self.out_tiles[b])

    def map_ops(self):
        dedicated_core = self.wafer.get_core(self.src, 0)
        for b in self.in_tiles:
            self.tile_ops[b].map_to_core(dedicated_core)
            self.stats.merge(self.tile_ops[b].stats)

    def calc_expected(self):
        expected = {"multicast": eval("*".join(map(str, self.dims))) * dtype_to_byte(self.input_tensor.prec)}
        expected["reads"] = 0
        expected["writes"] = 0
        return expected

    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.uid, expected=expected, dims=self.dims, tile_size=self.tile_size)

if __name__ == "__main__":
    from core_level.common.wafer import Wafer
    from core_level.common.tensor import reset_tensor_registry
    from core_level.common.graph import Graph

    reset_tensor_registry()

    node_grid = (2, 2)
    core_grid = (4, 4)

    wafer = Wafer(node_grid, core_grid)

    src_id = 0
    dst_ids = [2, 3]

    ops = {}
    ops[src_id] = {}
    op_id = f"multicast_0"
    ops[src_id][op_id] = {
        "type": "Multicast",
        "inputs": [f"{src_id}:input_tensor"],
        "outputs": [f"{dst}:output_tensor" for dst in dst_ids]
    }
    
    graph = Graph(iter=0, num_nodes=wafer.num_nodes, ops=ops)

    dims = [32, 32]
    tile_size = [16, 16]

    input_tensor = Tensor(
        uid=f"{src_id}:input_tensor",
        dims=dims,
        prec="fp16",
    )
    input_tensor.map_to_memory(wafer.banks[src_id], tile_size=tile_size, addr_offset=0)

    layer = MulticastLayer(f"multicast_0", src_id, dst_ids, graph, dims, wafer, "fp16")

    traces = wafer.get_traces()
    for node_id in traces:
        print("\n=== Node {} Traces ===".format(node_id))
        for core_id in traces[node_id]:
            print("-- Core {} --".format(core_id))
            for inst in traces[node_id][core_id]:
                print(inst)