from dpl.core import Variable, UnaryFunction, ndarray


class Reshape(UnaryFunction):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        gx = reshape(gy, self.x_shape)
        return gx


def reshape(self: Variable, shape: tuple[int, ...]) -> Variable:
    return Reshape(shape).apply(self)
