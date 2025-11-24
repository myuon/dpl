import numpy as np
from dpl.core import Variable, Function, as_variable
from dpl import metal
import jax.numpy as jnp


class BroadcastTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
        (x,) = xs
        self.x_shape = x.shape
        xp = metal.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        gx = sum_to(gy, self.x_shape)
        return gx


class SumTo(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        (x,) = xs
        self.x_shape = x.shape
        y = _sum_to(x, self.shape)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        gx = broadcast_to(gy, self.x_shape)
        return gx


def _sum_to(x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum elements along axes to output an array of a given shape."""
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def sum_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return x

    return SumTo(shape).apply(x)


def broadcast_to(x: Variable, shape: tuple[int, ...]) -> Variable:
    if x.shape == shape:
        return x

    return BroadcastTo(shape).apply(x)
