import numpy as np
from dpl.core import Variable, Function


class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        (x,) = xs
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        gx = reshape(gy, self.x_shape)
        return gx


def reshape(self: Variable, shape: tuple[int, ...]) -> Variable:
    return Reshape(shape).apply(self)
