
import logging

from typing import List
from src.node_level.common.utils import dtype_to_byte, get_dict_val, set_dict_val
import itertools

from src.core_level.common.stats import Stats
from src.core_level.common.isa import InstructionSet
from src.core_level.common.tensor import Tensor

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
                assert len(send_mem_sizes) == len(recv_mem_sizes), "Mismatched number of memory banks between send0 and next tile in TileMulticastOp {}.".format(self.id)

                recv_bank = list(recv_mem_sizes.keys())[i]
                recv_size = recv_mem_sizes[recv_bank]

                assert send_size == recv_size, "Mismatched send0 and next tile sizes in TileMulticastOp {}.".format(self.id)
                
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
        wafer: wafer that the layer is mapped to
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

        assert len(dims) in [1, 2, 3], "Multicast operation supports only 1D, 2D, or 3D tensors."

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=dims,
            prec=self.prec
        )
        assert self.input_tensor.tile_size is not None, "Input tensor {} of Multicast operation {} on node {} does not have tile size.".format(self.input_tensor.uid, uid, src)
        self.tile_size = list(self.input_tensor.tile_size)

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
        def _create1d(self):
            D0 = self.dims
            T0 = self.input_tensor.tile_size

            for i0, p0 in enumerate(range(0, D0, T0)):
                tiled_0 = min(T0, D0 - p0)

                self.in_tiles[i0] = self.input_tensor.slice([(i0*T0, i0*T0 + tiled_0),])

                self.out_tiles[i0] = []
                for d, dst_node in enumerate(self.dsts):
                    self.out_tiles[i0].append(self.output_tensors[d].slice([(i0*T0, i0*T0 + tiled_0),]))

        def _create2d(self):
            D0, D1 = self.dims
            T0, T1 = self.input_tensor.tile_size

            for i0, p0 in enumerate(range(0, D0, T0)):
                self.in_tiles[i0] = {}
                self.out_tiles[i0] = {}
                for i1, p1 in enumerate(range(0, D1, T1)):
                    tiled_0 = min(T0, D0 - p0)
                    tiled_1 = min(T1, D1 - p1)

                    self.in_tiles[i0][i1] = self.input_tensor.slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1)])

                    self.out_tiles[i0][i1] = []
                    for d, dst_node in enumerate(self.dsts):
                        self.out_tiles[i0][i1].append(self.output_tensors[d].slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1)]))

        def _create3d(self):
            D0, D1, D2 = self.dims
            T0, T1, T2 = self.input_tensor.tile_size

            for i0, p0 in enumerate(range(0, D0, T0)):
                self.in_tiles[i0] = {}
                self.out_tiles[i0] = {}
                for i1, p1 in enumerate(range(0, D1, T1)):
                    self.in_tiles[i0][i1] = {}
                    self.out_tiles[i0][i1] = {}
                    for i2, p2 in enumerate(range(0, D2, T2)):
                        tiled_0 = min(T0, D0 - p0)
                        tiled_1 = min(T1, D1 - p1)
                        tiled_2 = min(T2, D2 - p2)

                        self.in_tiles[i0][i1][i2] = self.input_tensor.slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1), (i2*T2, i2*T2 + tiled_2)])

                        self.out_tiles[i0][i1][i2] = []
                        for d, dst_node in enumerate(self.dsts):
                            self.out_tiles[i0][i1][i2].append(self.output_tensors[d].slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1), (i2*T2, i2*T2 + tiled_2)]))

        for d, dst in enumerate(self.dsts):
            self.output_tensors[d].map_to_memory(self.wafer.banks[dst], tile_size=self.tile_size, addr_offset=0)

        if len(self.dims) == 1:
            _create1d(self)
        elif len(self.dims) == 2:
            _create2d(self)
        elif len(self.dims) == 3:
            _create3d(self)
        else:
            raise NotImplementedError("MulticastLayer.create_tiles() only supports 1D, 2D and 3D tensors.")

    def create_ops(self):
        indices = []
        tmp_ops = self.in_tiles
        for i in range(len(self.dims)):
            indices.append(list(tmp_ops.keys()))
            tmp_ops = tmp_ops[0]
        indices = list(itertools.product(*indices))

        for ind in indices:
            in_tiles = get_dict_val(self.in_tiles, ind)
            out_tiles = get_dict_val(self.out_tiles, ind)
            op = TileMulticastOp(
                "{}_multicast_{}".format(self.uid, "_".join(map(str, ind))), 
                in_tiles, 
                out_tiles
            )
            set_dict_val(self.tile_ops, ind, op)

    def map_ops(self):        
        dedicated_core = self.wafer.get_core(self.src, 0)

        indices = []
        tmp_ops = self.tile_ops
        for i in range(len(self.dims)):
            indices.append(list(tmp_ops.keys()))
            tmp_ops = tmp_ops[0]
        indices = list(itertools.product(*indices))

        for ind in indices:
            op = get_dict_val(self.tile_ops, ind)
            op.map_to_core(dedicated_core)
            self.stats.merge(op.stats)

    def calc_expected(self):
        expected = {"multicast": eval("*".join(map(str, self.dims))) * dtype_to_byte(self.input_tensor.prec)}
        expected["reads"] = 0
        expected["writes"] = 0
        return expected

    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.src, expected=expected, dims=self.dims, tile_size=self.tile_size)

if __name__ == "__main__":
    from src.core_level.common.wafer import Wafer
    from src.core_level.common.tensor import reset_tensor_registry
    from src.core_level.common.graph import Graph

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