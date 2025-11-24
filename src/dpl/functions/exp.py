from dpl.core import Variable, Function, ndarray, get_array_module


class Exp(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

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
