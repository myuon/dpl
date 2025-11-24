from dpl.core import Variable, Function, ndarray
from dpl.core import metal


class Log(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        xp = metal.get_array_module(x)
        x_clipped = xp.clip(x, 1e-15, None)
        y = xp.log(x_clipped)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = gy / x
        return gx


def log(self: Variable) -> Variable:
    return Log().apply(self)
