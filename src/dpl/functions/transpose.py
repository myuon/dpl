from dpl.core import Variable, Function, ndarray, get_array_module


class Transpose(Function):
    def __init__(self, *axes: int) -> None:
        self.axes = axes

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        return x.transpose(*self.axes)

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        if self.axes is None or len(self.axes) == 0:
            return transpose(gy)

        axes_len = len(self.axes)
        xp = get_array_module(gy.data)
        inv_axes = tuple(xp.argsort([ax % axes_len for ax in self.axes]))
        gx = transpose(gy, *inv_axes)
        return gx


def transpose(self: Variable, *axes: int) -> Variable:
    return Transpose(*axes).apply(self)
