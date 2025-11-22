import numpy as np


def as_nparray(x):
    if np.isscalar(x):
        return np.array(x)

    return x


def unwrap[T](x: T | None) -> T:
    assert x is not None
    return x
