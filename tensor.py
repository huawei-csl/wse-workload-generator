



def concat(tensors, axis):
    if axis < 0:
        axis += len(tensors[0].dims)

    new_dims = list(tensors[0].dims)

    for dim in range(len(new_dims)):
        if dim != axis:
            for tensor in tensors[1:]:
                assert new_dims[dim] == tensor.dims[dim], "Dimensions other than concatenation axis must match"

    new_dims[axis] = sum(tensor.dims[axis] for tensor in tensors)
    return Tensor(tensors[0].uid + "_concat", tuple(new_dims))

class Tensor:
    def __init__(self, uid, dims) -> None:
        self.uid = uid
        self.dims = dims

    def view(self, new_dims):
        assert eval("*".join([str(d) for d in new_dims])) == eval("*".join([str(d) for d in self.dims])), "Total number of elements must remain the same in view operation"
        return Tensor(self.uid + "_view", new_dims)

    def split(self, split_dims, axis):
        if axis < 0:
            axis += len(self.dims)

        new_dims = list(self.dims)
        assert new_dims[axis] == sum(split_dims), "Sum of split dimensions must equal the original dimension size"
        return [Tensor(self.uid + f"_split_{i}", tuple(new_dims[:axis] + [split_dims[i]] + new_dims[axis+1:])) for i in range(len(split_dims))]

    def slice(self, indices, axis, uid=None):
        new_dims = list(self.dims)
        new_dims[axis] = len(indices)

        if uid is None:
            uid = self.uid + "_slice"
        return Tensor(uid, tuple(new_dims))
    
    def numel(self):
        return eval("*".join([str(d) for d in self.dims]))
        