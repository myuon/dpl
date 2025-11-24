from dpl.core import Variable, UnaryFunction, get_array_module


class Tanh(UnaryFunction):
    def forward(self, *xs):
        (x,) = xs
        xp = get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = gy * (1 - x**2)
        return gx


def tanh(x):
    return Tanh().apply(x)
