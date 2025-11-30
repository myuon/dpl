from dpl.core import Variable, UnaryFunction, get_array_module, unwrap


class Tanh(UnaryFunction):
    def forward(self, *xs):
        (x,) = xs
        xp = get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (y,) = self.outputs
        y0 = unwrap(y())
        gx = gy * (1 - y0 * y0)
        return gx


def tanh(x):
    return Tanh().apply(x)
