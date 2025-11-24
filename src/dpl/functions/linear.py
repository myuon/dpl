from dpl.core import Variable, as_variable, ndarray
from dpl.functions.matmul import matmul


def linear(
    x: Variable | ndarray, W: Variable | ndarray, b: Variable | None = None
) -> Variable:
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None

    return y
