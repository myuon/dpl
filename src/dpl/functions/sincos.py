from dpl import Variable, UnaryFunction, ndarray, get_array_module


class Sin(UnaryFunction):
    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        xp = get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


def sin(
    self: Variable,
) -> Variable:
    return Sin().apply(self)


class Cos(UnaryFunction):
    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        xp = get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(
    self: Variable,
) -> Variable:
    return Cos().apply(self)
