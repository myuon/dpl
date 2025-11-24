from dpl.core import Variable, UnaryFunction, ndarray, get_array_module


class Exp(UnaryFunction):
    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        xp = get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = exp(x) * gy
        return gx


def exp(self: Variable) -> Variable:
    return Exp().apply(self)
