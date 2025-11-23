import numpy as np
from dpl import Variable, as_variable, as_nparray


def accuracy(y_pred: Variable, t: Variable) -> Variable:
    pred = y_pred.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data).astype(np.float32)
    acc = result.mean()
    return as_variable(as_nparray(acc))
