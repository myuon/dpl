from dpl.core.function import Function
from dpl.core.utils import as_nparray
from dpl.core.variable import Variable
import numpy as np


class Add(Function):
    def apply(self, x0: Variable, x1: Variable | np.ndarray) -> Variable:
        result = super().__call__(x0, x1)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        from dpl.functions.broadcast_to import sum_to

        (gy,) = gys
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


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

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        (gy,) = gys
        x0, x1 = self.inputs
        return gy * x1, gy * x0


def mul(
    self: Variable, other: Variable | int | float | np.ndarray | np.number
) -> Variable:
    return Mul().apply(self, as_nparray(other))


class Neg(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        (x,) = xs
        return -x

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        return -gy


def neg(self: Variable) -> Variable:
    return Neg().apply(self)


class Sub(Function):
    def apply(self, x0: Variable | np.ndarray, x1: Variable | np.ndarray) -> Variable:
        result = super().__call__(x0, x1)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 - x1
        return y

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        (gy,) = gys
        return gy, -gy


def sub(
    self: Variable, other: Variable | int | float | np.ndarray | np.number
) -> Variable:
    return Sub().apply(self, as_nparray(other))


def rsub(
    self: Variable, other: Variable | int | float | np.ndarray | np.number
) -> Variable:
    return Sub().apply(as_nparray(other), self)


class Div(Function):
    def apply(self, x0: Variable | np.ndarray, x1: Variable | np.ndarray) -> Variable:
        result = super().__call__(x0, x1)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        x0, x1 = xs
        y = x0 / x1
        return y

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        (gy,) = gys
        (x0, x1) = self.inputs
        return gy / x1, -gy * x0 / (x1**2)


def div(
    self: Variable, other: Variable | int | float | np.ndarray | np.number
) -> Variable:
    return Div().apply(self, as_nparray(other))


def rdiv(
    self: Variable, other: Variable | int | float | np.ndarray | np.number
) -> Variable:
    return Div().apply(as_nparray(other), self)


class Pow(Function):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        (x,) = xs
        return x**self.exponent

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = self.exponent * (x ** (self.exponent - 1)) * gy
        return gx


def pow(self: Variable, exponent: float) -> Variable:
    return Pow(exponent).apply(self)


def load_operators() -> None:
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
