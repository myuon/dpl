from dpl.core import Variable, BinaryFunction, ndarray


class MatMul(BinaryFunction):
    def forward(self, *xs: ndarray) -> ndarray:
        x, W = xs
        y = x.dot(W)
        return y

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        (gy,) = gys
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(self: Variable, W: Variable) -> Variable:
    return MatMul().apply(self, W)
