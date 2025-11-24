from dpl.core import Variable, as_variable, ndarray
from dpl.functions.exp import exp


def sigmoid(x: Variable | ndarray) -> Variable:
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y
