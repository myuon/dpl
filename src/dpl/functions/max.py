from dpl.core import Variable, UnaryFunction, as_variable, ndarray, get_array_module


def reshape_max_backward(
    gy: ndarray, x_shape: tuple[int, ...], axis: int | None, keepdims: bool
) -> ndarray:
    """Reshape gradient for max backward pass.

    When keepdims=False, max removes dimensions, so we need to restore them
    for broadcasting in the backward pass.
    """
    if keepdims:
        # Shape is already correct
        return gy

    if axis is None:
        # All dimensions were reduced, restore to shape with all 1s
        return gy.reshape((1,) * len(x_shape))

    # Single axis was reduced, restore that dimension
    # Handle negative axis
    ndim = len(x_shape)
    if axis < 0:
        axis = axis + ndim

    # Insert dimension at the axis position
    shape = list(gy.shape)
    shape.insert(axis, 1)
    return gy.reshape(tuple(shape))


class Max(UnaryFunction):
    def __init__(self, axis: int | None = None, keepdims: bool = False) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        self.x_shape = x.shape
        self.x = x
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        gy_reshaped = reshape_max_backward(gy.data, self.x_shape, self.axis, self.keepdims)
        xp = get_array_module(self.x)

        # Create mask where x equals max value
        max_val = self.x.max(axis=self.axis, keepdims=True)
        mask = (self.x == max_val).astype(self.x.dtype)

        # Normalize by the number of max elements (to handle ties)
        num_max = mask.sum(axis=self.axis, keepdims=True)
        mask = mask / num_max

        # Broadcast gradient and multiply by mask
        gy_broadcast = xp.broadcast_to(gy_reshaped, self.x_shape)
        gx = gy_broadcast * mask

        return as_variable(gx)


def max(self: Variable, axis: int | None = None, keepdims: bool = False) -> Variable:
    return Max(axis, keepdims).apply(self)
