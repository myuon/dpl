from dpl import Variable, ndarray, get_array_module
from dpl.core import BinaryFunction
import dpl.functions as F


def softmax(x: Variable, axis: int = 1) -> Variable:
    y = F.exp(x)
    sum_y = F.sum(y, axis=axis, keepdims=True)
    return y / sum_y


class SoftmaxCrossEntropy(BinaryFunction):
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

        N, CLS_NUM = x.shape

        gy *= 1.0 / N
        y = softmax(x)

        xp = get_array_module(x.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data.astype(int)]
        y = (y - t_onehot) * gy

        return y, Variable(xp.zeros(t.shape))


def softmax_cross_entropy(x: Variable, t: Variable) -> Variable:
    return SoftmaxCrossEntropy().apply(x, t)
