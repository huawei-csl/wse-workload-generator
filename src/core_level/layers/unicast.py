import logging
import itertools

from typing import List

from src.node_level.common.utils import dtype_to_byte, get_dict_val, set_dict_val
from src.core_level.common.stats import Stats
from src.core_level.common.isa import InstructionSet
from src.core_level.common.tensor import Tensor

class TileUnicastOp:
    def __init__(self, id, input_tile, out_tile) -> None:
        self.id = id
        self.input_tile = input_tile
        self.out_tile = out_tile
        self.mapped_core = None
        self.stats = Stats()
        logging.debug("TileUnicastOp {} is created with input tile {}, out tile {}.".format(self.id, self.input_tile.id, self.out_tile.id))

    def map_to_core(self, core: "Core"):
        assert self.mapped_core is None, "TileUnicastOp {} is already mapped to core {}.".format(self.id, self.mapped_core.core_id)
        self.mapped_core = core
        core.add_instruction(self)
        logging.debug("TileUnicastOp {} is mapped to core {}.".format(self.id, self.mapped_core.core_id))

    def get_traces(self) -> List[str]:
        traces = []

        send_mem_sizes = self.input_tile.get_physical_address()
        recv_mem_sizes = self.out_tile.get_physical_address()

        assert sum(send_mem_sizes.values()) == sum(recv_mem_sizes.values()), "Mismatched total memory sizes between send0 and next tile in TileUnicastOp {}.".format(self.id)
        assert len(send_mem_sizes) == 1 or len(recv_mem_sizes) == 1, "TileUnicastOp {} supports only 1-to-many or many-to-1 mem copies.".format(self.id)

        total_data_size = 0
        if len(recv_mem_sizes) > 1:
            for i in range(len(recv_mem_sizes)):
                send_bank = list(send_mem_sizes.keys())[0]

                recv_bank = list(recv_mem_sizes.keys())[i]
                data_size = recv_mem_sizes[recv_bank]
                
                if send_bank != recv_bank:
                    traces.append(InstructionSet.COPY(send_bank.bank_id, recv_bank.bank_id, data_size, self.id))
                    self.stats.add_copy(data_size)
                total_data_size += data_size
        else:
            for i in range(len(send_mem_sizes)):
                send_bank = list(send_mem_sizes.keys())[i]
                data_size = send_mem_sizes[send_bank]

                recv_bank = list(recv_mem_sizes.keys())[0]
                
                if send_bank != recv_bank:
                    traces.append(InstructionSet.COPY(send_bank.bank_id, recv_bank.bank_id, data_size, self.id))
                    self.stats.add_copy(data_size)
                total_data_size += data_size
                
        assert total_data_size == sum(send_mem_sizes.values()), "Mismatched total data sizes in TileUnicastOp {}.".format(self.id)
        return traces

class UnicastLayer:
    ''' Unicast Layer.
    Args:
        uid: unique identifier for the layer
        src: source node index
        dst: destination node index
        dims: dimensions of the vector to be unicasted
        cores: list of Core objects available for mapping operations
        banks: list of MemoryBank objects available for mapping tiles
        prec: precision of the data (e.g., "fp16", "fp8")
    '''
    def __init__(self, uid, node_id, src, dst, graph, dims, wafer, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.src = src
        self.dst = dst
        self.dims = dims
        self.wafer = wafer 
        self.prec = prec

        self.graph_op = graph.get_op(src, uid)

        self.input_tensor = Tensor(
            uid=self.graph_op["inputs"][0],
            dims=dims,
            prec=self.prec,
        )
        assert self.input_tensor.tile_size is not None, "Input tensor {} of Multicast operation {} on node {} does not have tile size.".format(self.input_tensor.uid, uid, src)
        self.tile_size = list(self.input_tensor.tile_size)

        self.output_tensor = Tensor(
            uid=self.graph_op["outputs"][0],
            dims=dims,
            prec=self.prec,
        )

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
                self.out_tiles[i0] = self.output_tensor.slice([(i0*T0, i0*T0 + tiled_0),])

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
                    self.out_tiles[i0][i1] = self.output_tensor.slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1)])

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
                        self.out_tiles[i0][i1][i2] = self.output_tensor.slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1), (i2*T2, i2*T2 + tiled_2)])

        self.output_tensor.map_to_memory(self.wafer.banks[self.dst], tile_size=self.tile_size, addr_offset=0)

        if len(self.dims) == 1:
            _create1d(self)
        elif len(self.dims) == 2:
            _create2d(self)
        elif len(self.dims) == 3:
            _create3d(self)
        else:
            raise NotImplementedError("MulticastLayer.create_tiles() only supports 1D, 2D and 3D tensors.")

        # # Tile from last dimension
        # for d0, pD0 in enumerate(range(0, self.dims[-1], self.tile_size[-1])):
        #     tiled_B = min(self.tile_size[-1], self.dims[-1] - pD0)

        #     slice_indices = [(0, d) for d in self.dims[0:-1]] + [(pD0, pD0 + tiled_B)]

        #     self.in_tiles[d0] = self.input_tensor.slice(slice_indices)
        #     self.out_tiles[d0] = self.output_tensor.slice(slice_indices)

        #     send_mem_sizes = self.in_tiles[d0].get_physical_address()
        #     recv_mem_sizes = self.out_tiles[d0].get_physical_address()
        #     assert sum(send_mem_sizes.values()) == sum(recv_mem_sizes.values()), "Mismatched total memory sizes between send0 and next tile in TileUnicastOp {}.".format(self.id)

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
            op = TileUnicastOp(
                "{}_unicast_{}".format(self.uid, "_".join(map(str, ind))), 
                in_tiles, 
                out_tiles
            )
            set_dict_val(self.tile_ops, ind, op)

        # for b in self.in_tiles:
        #     self.tile_ops[b] = TileUnicastOp("{}_unicast_{}".format(self.uid, b), self.in_tiles[b], self.out_tiles[b])

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

        # dedicated_core = self.wafer.get_core(self.src, 0)
        # for b in self.tile_ops:
        #     self.tile_ops[b].map_to_core(dedicated_core)
        #     self.stats.merge(self.tile_ops[b].stats)

    def calc_expected(self):
        expected = {"copy": eval("*".join(map(str, self.dims))) * dtype_to_byte(self.input_tensor.prec)}
        expected["reads"] = 0
        expected["writes"] = 0
        return expected

    def log_stats(self):
        expected = self.calc_expected()
        self.stats.log_stats(self.uid, self.__class__.__name__, self.src, expected=expected, dims=self.dims, tile_size=self.tile_size)
