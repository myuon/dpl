import numpy as np
from dpl.core import Variable, Function


class Exp(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        (x,) = xs
        y = np.exp(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = exp(x) * gy
        return gx


def exp(self: Variable) -> Variable:
    return Exp().apply(self)
