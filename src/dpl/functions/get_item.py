import numpy as np
from dpl.core import Variable, Function, ndarray


class GetItemGrad(Function):
    def __init__(self, slices: tuple | int, input_shape: tuple) -> None:
        self.slices = slices
        self.input_shape = input_shape

    def apply(self, gy: Variable) -> Variable:
        result = super().__call__(gy)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        (gy,) = xs
        gx = np.zeros(self.input_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, *gys: Variable) -> Variable:
        (ggx,) = gys
        return get_item(ggx, self.slices)


class GetItem(Function):
    def __init__(self, slices: tuple | int) -> None:
        self.slices = slices

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        y = x[self.slices]
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        f = GetItemGrad(self.slices, x.shape)

        return f.apply(gy)


def get_item(self: Variable, slices: tuple | int) -> Variable:
    return GetItem(slices).apply(self)
