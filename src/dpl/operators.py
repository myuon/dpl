from function import Function
from utils import as_nparray
from variable import Variable
import numpy as np


class Add(Function):
    def apply(self, x0: Variable, x1: Variable | np.ndarray) -> Variable:
        result = super().__call__(x0, x1)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 + x1
        return y

    def backward(self, *gys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        (gy,) = gys
        return gy, gy


def add(
    self: Variable, other: Variable | int | float | np.ndarray | np.number
) -> Variable:
    return Add().apply(self, as_nparray(other))


class Mul(Function):
    def apply(self, x0: Variable, x1: Variable | np.ndarray) -> Variable:
        result = super().__call__(x0, x1)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 * x1
        return y

    def backward(self, *gys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        (gy,) = gys
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return gy * x1, gy * x0


def mul(
    self: Variable, other: Variable | int | float | np.ndarray | np.number
) -> Variable:
    return Mul().apply(self, as_nparray(other))


class Square(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        (x,) = xs
        return x**2

    def backward(self, *gys: np.ndarray) -> np.ndarray:
        (gy,) = gys
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
