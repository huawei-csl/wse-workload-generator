
import compute_graph
from compute_graph import get_compute_graph

class TensorRegistry:
    _registry = {}

    @classmethod
    def register(cls, tensor):
        assert tensor.uid not in cls._registry, f"Tensor with uid {tensor.uid} is already registered."
        cls._registry[tensor.uid] = tensor

    @classmethod
    def get(cls, uid):
        return cls._registry.get(uid, None)

    @classmethod
    def is_exist(cls, uid):
        return uid in cls._registry

def get_tensor(uid):
    tensor = TensorRegistry.get(uid)
    assert tensor is not None, f"Tensor with uid {uid} not found."
    return tensor

class Tensor:
    def __new__(cls, uid, dims):
        if TensorRegistry.is_exist(uid):
            return TensorRegistry.get(uid)
        return super().__new__(cls)

    def __init__(self, uid, dims) -> None:
        if not hasattr(self, 'uid'):
            self.uid = uid
            self.dims = dims
            TensorRegistry.register(self)

    # def view(self, new_dims):
    #     assert eval("*".join([str(d) for d in new_dims])) == eval("*".join([str(d) for d in self.dims])), "Total number of elements must remain the same in view operation"
    #     return Tensor(self.uid + "_view", new_dims)

    # def split(self, split_dims, axis):
    #     if axis < 0:
    #         axis += len(self.dims)

    #     new_dims = list(self.dims)
    #     assert new_dims[axis] == sum(split_dims), "Sum of split dimensions must equal the original dimension size"
    #     return [Tensor(self.uid + f"_split_{i}", tuple(new_dims[:axis] + [split_dims[i]] + new_dims[axis+1:])) for i in range(len(split_dims))]

    def slice(self, indices, axis, uid=None):
        def convert_to_range(indices):
            if len(indices) == 1:
                return (indices[0], indices[0]+1)
            step = indices[1] - indices[0]
            for i in range(2, len(indices)):
                if indices[i] - indices[i-1] != step:
                    return None
            return (indices[0], indices[-1]+1)
        
        assert len(indices) > 0, "Indices list cannot be empty"
        assert axis >= 0 and axis < len(self.dims), "Axis out of bounds"

        index_rng = convert_to_range(indices)
        assert index_rng is not None, "Indices must form a valid range with consistent step size"

        new_dims = list(self.dims)
        new_dims[axis] = len(indices)

        if uid is None:
            if len(indices) == 1:
                uid = self.uid + f"_slice{index_rng[0]}"
            else:
                uid = self.uid + f"_slice{index_rng[0]}:{index_rng[1]}"
        return Tensor(uid, tuple(new_dims))
    
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

        self.output_tensor = Tensor(self.uid, new_dims)

    def forward(self, stats):
        stats.append(self.uid, "View", 0, 0, 0, 0, comm_group=None, dims=self.new_dims)
        get_compute_graph().add_node(self, [self.input_tensor], [self.output_tensor], attrs=None)        
        return self.output_tensor
    
class Split:
    def __init__(self, input_tensor, split_dims, axis, uid=None) -> None:
        if uid is None:
            self.uid = input_tensor.uid + f"_split"
        else:
            self.uid = uid

        self.input_tensor = input_tensor
        self.dims = list(input_tensor.dims)

        if axis < 0:
            axis += len(self.dims)

        assert len(split_dims) == 2, "Only support splitting into two tensors for now" 
        assert sum(split_dims) == input_tensor.dims[axis], "Sum of split dimensions must equal the original dimension size"

        self.output_tensors = [
            Tensor(self.uid + f"{0}:{split_dims[0]}", tuple(self.dims[:axis] + [split_dims[0]] + self.dims[axis+1:])),
            Tensor(self.uid + f"{split_dims[0]}:{split_dims[0]+split_dims[1]}", tuple(self.dims[:axis] + [split_dims[1]] + self.dims[axis+1:])),
        ]

    def forward(self, stats):
        stats.append(self.uid, "Split", 0, 0, 0, 0, comm_group=None, dims=self.dims)
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

        self.output_tensor = Tensor(self.uid, tuple(self.new_dims))

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

        self.input_tensors = input_tensors

        if axis < 0:
            axis += len(input_tensors[0].dims)

        self.new_dims = list(input_tensors[0].dims)

        for dim in range(len(self.new_dims)):
            if dim != axis:
                for tensor in input_tensors[1:]:
                    assert self.new_dims[dim] == tensor.dims[dim], "Dimensions other than concatenation axis must match"

        self.new_dims[axis] = sum(tensor.dims[axis] for tensor in input_tensors)
        self.output_tensor = Tensor(self.uid, tuple(self.new_dims))

    def forward(self, stats):
        stats.append(self.uid, "Concat", 0, 0, 0, 0, comm_group=None, dims=self.new_dims)
        get_compute_graph().add_node(self, [tensor for tensor in self.input_tensors], [self.output_tensor], attrs=None)        
        return self.output_tensor
    
class NoOp:
    def __init__(self, input_tensor, uid=None) -> None:
        if uid is None:
            self.uid = input_tensor.uid
        else:
            self.uid = uid

        self.input_tensor = input_tensor
        self.dims = input_tensor.dims
        self.output_tensor = Tensor(self.uid, self.dims)

    def forward(self, stats):
        stats.append(self.uid, "NoOp", 0, 0, 0, 0, comm_group=None, dims=self.dims)
        get_compute_graph().add_node(self, [self.input_tensor], [self.output_tensor], attrs=None)        
        return self.output_tensor