from dpl.core import Variable, UnaryFunction, as_variable, ndarray, get_array_module


def reshape_sum_backward(
    gy: ndarray, x_shape: tuple[int, ...], axis: int | None, keepdims: bool
) -> ndarray:
    """Reshape gradient for sum backward pass.

    When keepdims=False, sum removes dimensions, so we need to restore them
    for broadcasting in the backward pass.
    """
    if keepdims:
        # Shape is already correct
        return gy

    if axis is None:
        # All dimensions were summed, restore to shape with all 1s
        return gy.reshape((1,) * len(x_shape))

    # Single axis was summed, restore that dimension
    # Handle negative axis
    ndim = len(x_shape)
    if axis < 0:
        axis = axis + ndim

    # Insert dimension at the axis position
    shape = list(gy.shape)
    shape.insert(axis, 1)
    return gy.reshape(tuple(shape))


class Sum(UnaryFunction):
    def __init__(self, axis: int | None = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        gy = reshape_sum_backward(
            gy.data_required, self.x_shape, self.axis, self.keepdims
        )
        xp = get_array_module(gy)
        gx = xp.broadcast_to(gy, self.x_shape)
        return as_variable(gx)


def sum(self: Variable, axis: int | None = None, keepdims: bool = False) -> Variable:
    return Sum(axis, keepdims).apply(self)
