import numpy as np
from dpl.core import Variable, Function, as_variable
from dpl.functions.matmul import matmul


def linear(
    x: Variable | np.ndarray, W: Variable | np.ndarray, b: Variable | None = None
) -> Variable:
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None  # type: ignore

    return y
