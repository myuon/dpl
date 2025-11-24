from dpl.core import Variable, Function, ndarray


class MeanSquaredError(Function):
    def apply(self, y_pred: Variable, y_true: Variable) -> Variable:
        result = super().__call__(y_pred, y_true)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        x0, x1 = xs
        diff = x0 - x1
        return (diff**2).sum() / diff.size

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        from dpl.functions.broadcast_to import broadcast_to

        x0, x1 = self.inputs
        diff = x0 - x1
        (gy,) = gys
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2.0 / diff.size)
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(y_pred: Variable, y_true: Variable) -> Variable:
    return MeanSquaredError().apply(y_pred, y_true)
