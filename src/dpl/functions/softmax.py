import numpy as np
from dpl import Variable, ndarray
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
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Cross entropy
        log_p = np.log(np.clip(self.y, 1e-15, 1.0))
        t_onehot = np.eye(x.shape[1])[t.astype(int)]
        loss = -np.sum(t_onehot * log_p) / N

        return np.array(loss)

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        (gy,) = gys
        (x, t) = self.inputs

        N = x.shape[0]
        # Create one-hot encoding
        t_onehot = np.eye(x.shape[1])[t.data.astype(int)]

        # Gradient: (y - t_onehot) / N
        gx = (self.y - t_onehot) / N
        gx = Variable(gx * gy.data)  # Multiply by upstream gradient

        # No gradient for labels
        gt = Variable(np.zeros_like(t.data))

        return gx, gt


def softmax_cross_entropy(x: Variable, t: Variable) -> Variable:
    return SoftmaxCrossEntropy().apply(x, t)
