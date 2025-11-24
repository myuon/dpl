from dpl.core import Variable, UnaryFunction, ndarray, get_array_module


class Transpose(UnaryFunction):
    def __init__(self, *axes: int) -> None:
        self.axes = axes

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        return x.transpose(*self.axes)

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        if self.axes is None or len(self.axes) == 0:
            return transpose(gy)

        axes_len = len(self.axes)
        xp = get_array_module(gy.data)
        inv_axes = tuple(xp.argsort(xp.array([ax % axes_len for ax in self.axes])))
        gx = transpose(gy, *inv_axes)
        return gx


def transpose(self: Variable, *axes: int) -> Variable:
    return Transpose(*axes).apply(self)
