
import compute_graph
from compute_graph import get_compute_graph
import logging
from typing import List 

class TensorRegistry:
    _registry = {}

    @classmethod
    def register(cls, tensor):
        assert tensor.uid not in cls._registry, f"Tensor with uid {tensor.uid} is already registered."
        if tensor.node_id not in cls._registry:
            cls._registry[tensor.node_id] = {}
        
        cls._registry[tensor.node_id][tensor.uid] = tensor

    @classmethod
    def get(cls, uid, node_id):
        if node_id not in cls._registry:
            return None
        return cls._registry[node_id].get(uid, None)

    @classmethod
    def is_exist(cls, uid, node_id):
        if node_id not in cls._registry:
            return False
        return uid in cls._registry

    @classmethod
    def reset(cls):
        cls._registry = {}

def reset_tensor_registry():
    TensorRegistry.reset()

def get_tensor(uid, node_id):
    tensor = TensorRegistry.get(uid, node_id)
    assert tensor is not None, f"Tensor with uid {uid} and with node id {node_id} not found."
    return tensor

class Tensor:
    def __new__(cls, uid, node_id, dims) -> 'Tensor':
        if TensorRegistry.is_exist(uid, node_id):
            existing_tensor = TensorRegistry.get(uid, node_id)
            assert existing_tensor.dims == list(dims), f"Tensor with uid {uid} and node_id {node_id} already exists with different dimensions. Existing dims: {existing_tensor.dims}, New dims: {dims}"
            logging.debug(f"Tensor {node_id}:{uid} already exists. Returning existing instance.")
            return existing_tensor
        return super().__new__(cls)

    def __init__(self, uid, node_id, dims: List[int]) -> None:
        if not hasattr(self, 'uid'):
            self.uid = uid
            self.graph_id = f"{node_id}:{uid}"
            self.node_id = node_id
            self.dims = list(dims)
            TensorRegistry.register(self)
            logging.debug(f"Tensor {node_id}:{uid} with dims {dims} is created.")

    def numel(self):
        return eval("*".join([str(d) for d in self.dims]))


class View:
    def __init__(self, input_tensor, new_dims, uid=None) -> None:
        if uid is None:
            self.uid = input_tensor.uid + "_view"
        else:
            self.uid = uid

        self.input_tensor = input_tensor
        self.new_dims = new_dims
        assert eval("*".join([str(d) for d in new_dims])) == eval("*".join([str(d) for d in input_tensor.dims])), "Total number of elements must remain the same in view operation"

        self.output_tensor = Tensor(self.uid, input_tensor.node_id, new_dims)

    def forward(self, stats):
        stats.append(self.uid, "View", 0, 0, 0, 0, comm_group=None, dims=f"{self.input_tensor.dims} -> {self.new_dims}")
        get_compute_graph().add_node(self, [self.input_tensor], [self.output_tensor], attrs=None)        
        return self.output_tensor

class Transpose:
    def __init__(self, input_tensor, trans_dims, uid=None) -> None:
        if uid is None:
            self.uid = input_tensor.uid + "_transpose"
        else:
            self.uid = uid

        self.input_tensor = input_tensor
        self.trans_dims = trans_dims
        new_dims = list(input_tensor.dims)
        new_dims[trans_dims[0]], new_dims[trans_dims[1]] = input_tensor.dims[trans_dims[1]], input_tensor.dims[trans_dims[0]]

        self.output_tensor = Tensor(self.uid, input_tensor.node_id, new_dims)

    def forward(self, stats):
        stats.append(self.uid, "Transpose", 0, 0, 0, 0, comm_group=None, dims=f"{self.trans_dims} -> {self.input_tensor.dims} -> {self.output_tensor.dims}")
        get_compute_graph().add_node(self, [self.input_tensor], [self.output_tensor], attrs=None)
        return self.output_tensor


class Split:
    def __init__(self, input_tensor, split_dims, axis, uid=None) -> None:
        if uid is None:
            self.uid = input_tensor.uid + f"_split"
        else:
            self.uid = uid

        self.input_tensor = input_tensor
        self.split_dims = split_dims
        self.axis = axis

        self.dims = list(input_tensor.dims)

        if axis < 0:
            axis += len(self.dims)

        assert len(split_dims) == 2, "Only support splitting into two tensors for now" 
        assert sum(split_dims) == input_tensor.dims[axis], "Sum of split dimensions must equal the original dimension size"

        self.output_tensors = [
            Tensor(f"{self.uid}{0}:{split_dims[0]}", input_tensor.node_id, list(self.dims[:axis] + [split_dims[0]] + self.dims[axis+1:])),
            Tensor(f"{self.uid}{split_dims[0]}:{split_dims[0]+split_dims[1]}", input_tensor.node_id, list(self.dims[:axis] + [split_dims[1]] + self.dims[axis+1:])),
        ]

    def forward(self, stats):
        stats.append(self.uid, "Split", 0, 0, 0, 0, comm_group=None, dims=f"{self.axis} -> {self.split_dims} -> {self.dims}")
        get_compute_graph().add_node(self, [self.input_tensor], [tensor for tensor in self.output_tensors], attrs=None)        
        return self.output_tensors

class Slice:
    def __init__(self, input_tensor, indices, axis, uid=None) -> None:
        def convert_to_range(indices):
            if len(indices) == 1:
                return (indices[0], indices[0]+1)
            step = indices[1] - indices[0]
            for i in range(2, len(indices)):
                if indices[i] - indices[i-1] != step:
                    return None
            return (indices[0], indices[-1]+1)

        assert len(indices) > 0, "Indices list cannot be empty"
        assert axis >= 0 and axis < len(input_tensor.dims), "Axis out of bounds"

        index_rng = convert_to_range(indices)
        assert index_rng is not None, "Indices must form a valid range with consistent step size"

        self.input_tensor = input_tensor

        self.new_dims = list(input_tensor.dims)
        self.new_dims[axis] = len(indices)

        if uid is None:
            if len(indices) == 1:
                self.uid = input_tensor.uid + f"_slice{index_rng[0]}"
            else:
                self.uid = input_tensor.uid + f"_slice{index_rng[0]}:{index_rng[1]}"
        else:
            self.uid = uid

        self.output_tensor = Tensor(self.uid, input_tensor.node_id, list(self.new_dims))

    def forward(self, stats):
        stats.append(self.uid, "Slice", 0, 0, 0, 0, comm_group=None, dims=self.new_dims)
        get_compute_graph().add_node(self, [self.input_tensor], [self.output_tensor], attrs=None)        
        return self.output_tensor
    
class Concat:
    def __init__(self, input_tensors, axis, uid=None) -> None:
        if uid is None:
            self.uid = input_tensors[0].uid + "_concat"
        else:
            self.uid = uid

        self.axis = axis

        # assert len(input_tensors) == 2, "Only support concatenating two tensors for now"

        node_id = input_tensors[0].node_id
        for tensor in input_tensors[1:]:
            assert tensor.node_id == node_id, "All input tensors must belong to the same node"
        
        self.input_tensors = input_tensors

        if axis < 0:
            axis += len(input_tensors[0].dims)

        self.new_dims = list(input_tensors[0].dims)

        for dim in range(len(self.new_dims)):
            if dim != axis:
                for tensor in input_tensors[1:]:
                    assert self.new_dims[dim] == tensor.dims[dim], "Dimensions other than concatenation axis must match"

        self.new_dims[axis] = sum(tensor.dims[axis] for tensor in input_tensors)
        self.output_tensor = Tensor(self.uid, node_id, list(self.new_dims))

    def forward(self, stats):
        stats.append(self.uid, "Concat", 0, 0, 0, 0, comm_group=None, dims=f"{self.axis} -> {[tensor.dims[self.axis] for tensor in self.input_tensors]} -> {self.new_dims}")
        get_compute_graph().add_node(self, [tensor for tensor in self.input_tensors], [self.output_tensor], attrs=None)        
        return self.output_tensor
