import numpy as np
from dpl.core import Variable, Function, ndarray


class Transpose(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        return x.T

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        gx = transpose(gy)
        return gx


def transpose(self: Variable) -> Variable:
    return Transpose().apply(self)
