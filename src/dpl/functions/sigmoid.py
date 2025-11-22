import numpy as np
from dpl.core import Variable, Function, as_variable
from dpl.functions.exp import exp


def sigmoid(x: Variable | np.ndarray) -> Variable:
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y
