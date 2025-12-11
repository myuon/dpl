from dpl.core import Variable, BinaryFunction, ndarray, get_array_module


class WeightSum(BinaryFunction):
    """
    Weighted sum for attention mechanism.

    Computes context vector as weighted sum of hidden states.
    c = sum_t(a_t * hs_t)
    """

    def forward(self, *xs: ndarray) -> ndarray:
        hs, a = xs
        # hs: (batch_size, seq_len, hidden_size)
        # a: (batch_size, seq_len)
        xp = get_array_module(hs)

        ar = a.reshape(a.shape[0], a.shape[1], 1)
        c = xp.sum(hs * ar, axis=1)
        return c

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        (dc,) = gys
        hs, a = self.inputs
        xp = get_array_module(dc.data)

        ar = a.data_required.reshape(a.shape[0], a.shape[1], 1)
        dcr = dc.data_required.reshape(dc.shape[0], 1, dc.shape[1])

        dhs = ar * dcr
        da = xp.sum(hs.data_required * dcr, axis=2)

        return Variable(dhs), Variable(da)


def weight_sum(hs: Variable, a: Variable) -> Variable:
    """Compute weighted sum of hidden states.

    Args:
        hs: Hidden states (batch_size, seq_len, hidden_size)
        a: Attention weights (batch_size, seq_len)

    Returns:
        Context vector (batch_size, hidden_size)
    """
    return WeightSum().apply(hs, a)
