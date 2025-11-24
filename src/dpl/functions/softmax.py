from dpl import Variable, ndarray, get_array_module
from dpl.core import Function
import dpl.functions as F


def softmax(x: Variable, axis: int = 1) -> Variable:
    y = F.exp(x)
    sum_y = F.sum(y, axis=axis, keepdims=True)
    return y / sum_y


class SoftmaxCrossEntropy(Function):
    def apply(self, x: Variable, t: Variable) -> Variable:
        result = super().__call__(x, t)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        x, t = xs
        N = x.shape[0]

        # Softmax (numerically stable)
        xp = get_array_module(x)
        x_max = xp.max(x, axis=1, keepdims=True)
        exp_x = xp.exp(x - x_max)
        self.y = exp_x / xp.sum(exp_x, axis=1, keepdims=True)

        # Cross entropy
        log_p = xp.log(xp.clip(self.y, 1e-15, 1.0))
        t_onehot = xp.eye(x.shape[1])[t.astype(int)]
        loss = -xp.sum(t_onehot * log_p) / N

        return xp.array(loss)

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        (gy,) = gys
        (x, t) = self.inputs

        N = x.shape[0]
        # Create one-hot encoding
        xp = get_array_module(x.data)
        t_onehot = xp.eye(x.shape[1])[t.data.astype(int)]

        # Gradient: (y - t_onehot) / N
        gx = (self.y - t_onehot) / N
        gx = Variable(gx * gy.data)  # Multiply by upstream gradient

        # No gradient for labels
        gt = Variable(xp.zeros_like(t.data))

        return gx, gt


def softmax_cross_entropy(x: Variable, t: Variable) -> Variable:
    return SoftmaxCrossEntropy().apply(x, t)
