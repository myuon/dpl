import numpy as np
from dpl.core import Variable, Function


class Log(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        (x,) = xs
        # Clip to avoid log(0)
        x_clipped = np.clip(x, 1e-15, None)
        y = np.log(x_clipped)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = gy / x
        return gx


def log(self: Variable) -> Variable:
    return Log().apply(self)
