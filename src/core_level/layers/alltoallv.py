import logging 
import itertools

from src.core_level.common.tensor import Tensor
from src.core_level.common.stats import Stats
from src.core_level.layers.unicast import TileUnicastOp
from src.node_level.common.utils import get_dict_val, set_dict_val

class AlltoAllv:
    def __init__(self, uid, node_id, comm_group, graph, dims, input_split, wafer, prec) -> None:
        self.uid = uid
        self.node_id = node_id
        self.comm_group = comm_group
        self.dims = dims
        self.input_split = input_split
        self.wafer = wafer 
        self.prec = prec

        self.graph_op = graph.get_op(node_id, uid)

        inputs = self.graph_op["inputs"]
        outputs = self.graph_op["outputs"]

        self.stats = Stats()
        
        assert len(inputs) == len(outputs) == len(input_split) == len(self.comm_group), "AlltoAllv operation {} on node {} must have number of inputs equal to the size of the communication group".format(uid, node_id)
        
        self.input_tensors = []
        tile_size = None
        for d, input_name in enumerate(inputs):
            dims = list(self.dims)
            dims[0] = input_split[d]
            input = Tensor(
                    uid=input_name,
                    dims=dims,
                    prec=self.prec,
                )
            self.input_tensors.append(input)
            if not input.is_empty():
                assert input.tile_size is not None, "Input tensor {} of Multicast operation {} on node {} does not have tile size.".format(input.uid, uid, node_id)
            
                if tile_size is None:
                    tile_size = list(input.tile_size)
                else:
                    assert tile_size == list(input.tile_size), "All input tensors of AlltoAllv operation {} on node {} must have the same tile size".format(uid, node_id)

        self.tile_size = tile_size

        self.output_tensors = []
        for d, output_name in enumerate(outputs):
            dims = list(self.dims)
            dims[0] = input_split[d]
            output = Tensor(
                uid=output_name,
                dims=dims,
                prec=self.prec,
            )
            self.output_tensors.append(output)
        
        for i in range(len(self.input_tensors)):
            assert self.input_tensors[i].dims == self.output_tensors[i].dims, "Input tensor {} and output tensor {} of AlltoAllv operation {} on node {} must have the same shape".format(self.input_tensors[i].uid, self.output_tensors[i].uid, uid, node_id)

        self.in_tiles = [{} for _ in range(len(self.input_tensors))]
        self.out_tiles = [{} for _ in range(len(self.output_tensors))]
        self.tile_ops = [{} for _ in range(len(self.output_tensors))]

        if sum(input_split) == 0:
            logging.debug("AlltoAllv operation {} on node {} does not send any data, skipping.".format(uid, node_id))
            return 

        self.create_tiles()
        self.create_ops()

        self.map_ops()

        return

    def create_tiles(self):
        def _create1d(self):
            for d in range(len(self.input_tensors)):
                if self.input_tensors[d].is_empty():
                    continue
                D0 = self.input_tensors[d].dims
                T0 = self.input_tensors[d].tile_size

                for i0, p0 in enumerate(range(0, D0, T0)):
                    tiled_0 = min(T0, D0 - p0)

                    self.in_tiles[d][i0] = self.input_tensors[d].slice([(i0*T0, i0*T0 + tiled_0),])
                    self.out_tiles[d][i0] = self.output_tensors[d].slice([(i0*T0, i0*T0 + tiled_0),])

        def _create2d(self):
            for d in range(len(self.input_tensors)):
                if self.input_tensors[d].is_empty():
                    continue
                D0, D1 = self.input_tensors[d].dims
                T0, T1 = self.input_tensors[d].tile_size

                for i0, p0 in enumerate(range(0, D0, T0)):
                    self.in_tiles[d][i0] = {}
                    self.out_tiles[d][i0] = {}
                    for i1, p1 in enumerate(range(0, D1, T1)):
                        tiled_0 = min(T0, D0 - p0)
                        tiled_1 = min(T1, D1 - p1)

                        self.in_tiles[d][i0][i1] = self.input_tensors[d].slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1)])
                        self.out_tiles[d][i0][i1] = self.output_tensor[d].slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1)])

        def _create3d(self):
            for d in range(len(self.input_tensors)):
                if self.input_tensors[d].is_empty():
                    continue
                D0, D1, D2 = self.input_tensors[d].dims
                T0, T1, T2 = self.input_tensors[d].tile_size

                for i0, p0 in enumerate(range(0, D0, T0)):
                    self.in_tiles[d][i0] = {}
                    self.out_tiles[d][i0] = {}
                    for i1, p1 in enumerate(range(0, D1, T1)):
                        self.in_tiles[d][i0][i1] = {}
                        self.out_tiles[d][i0][i1] = {}
                        for i2, p2 in enumerate(range(0, D2, T2)):
                            tiled_0 = min(T0, D0 - p0)
                            tiled_1 = min(T1, D1 - p1)
                            tiled_2 = min(T2, D2 - p2)

                            self.in_tiles[d][i0][i1][i2] = self.input_tensors[d].slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1), (i2*T2, i2*T2 + tiled_2)])
                            self.out_tiles[d][i0][i1][i2] = self.output_tensors[d].slice([(i0*T0, i0*T0 + tiled_0), (i1*T1, i1*T1 + tiled_1), (i2*T2, i2*T2 + tiled_2)])

        for d, dst in enumerate(self.comm_group):
            if self.output_tensors[d].is_empty():
                continue
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
        for d in range(len(self.input_tensors)):
            if self.input_tensors[d].is_empty():
                continue
            indices = []
            tmp_ops = self.in_tiles[d]
            for i in range(len(self.input_tensors[d].dims)):
                indices.append(list(tmp_ops.keys()))
                tmp_ops = tmp_ops[0]
            indices = list(itertools.product(*indices))

            for ind in indices:
                in_tiles = get_dict_val(self.in_tiles[d], ind)
                out_tiles = get_dict_val(self.out_tiles[d], ind)
                op = TileUnicastOp(
                    "{}_unicast_{}".format(self.uid, "_".join(map(str, ind))), 
                    in_tiles, 
                    out_tiles
                )
                set_dict_val(self.tile_ops[d], ind, op)

    def map_ops(self):
        dedicated_core = self.wafer.get_core(self.node_id, 0)

        for d in range(len(self.input_tensors)):
            if self.input_tensors[d].is_empty():
                continue
            indices = []
            tmp_ops = self.tile_ops[d]
            for i in range(len(self.input_tensors[d].dims)):
                indices.append(list(tmp_ops.keys()))
                tmp_ops = tmp_ops[0]
            indices = list(itertools.product(*indices))

            for ind in indices:
                op = get_dict_val(self.tile_ops[d], ind)
                op.map_to_core(dedicated_core)
                self.stats.merge(op.stats)

    def log_stats(self):
        expected = None
        self.stats.log_stats(self.uid, self.__class__.__name__, self.node_id, expected=expected, dims=self.dims, tile_size=self.tile_size)
