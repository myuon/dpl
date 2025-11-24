import numpy as np
from dpl import Function, Variable, as_variable


class ReLU(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        (x,) = xs
        y = np.maximum(0, x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (x,) = self.inputs
        (gy,) = gys
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x: Variable) -> Variable:
    return ReLU().apply(x)
