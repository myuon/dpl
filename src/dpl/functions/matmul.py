import numpy as np
from dpl.core import Variable, Function


class MatMul(Function):
    def apply(self, x0: Variable, x1: Variable) -> Variable:
        result = super().__call__(x0, x1)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x, W = xs
        y = x.dot(W)
        return y

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        (gy,) = gys
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(self: Variable, W: Variable) -> Variable:
    return MatMul().apply(self, W)
