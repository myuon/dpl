import numpy as np
from dpl import Variable, Function, ndarray, get_array_module


class Sin(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        xp = get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


def sin(
    self: Variable,
) -> Variable:
    return Sin().apply(self)


class Cos(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        y = np.cos(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(
    self: Variable,
) -> Variable:
    return Cos().apply(self)
