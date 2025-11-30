from dpl.core import Variable, UnaryFunction, ndarray, get_array_module


class Sqrt(UnaryFunction):
    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        xp = get_array_module(x)
        y = xp.sqrt(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        # d(sqrt(x))/dx = 1/(2*sqrt(x))
        gx = gy / (2 * sqrt(x))
        return gx


def sqrt(self: Variable) -> Variable:
    return Sqrt().apply(self)
