import numpy as np


def as_nparray(x, array_module=np) -> np.ndarray:
    if np.isscalar(x):
        return array_module.array(x)

    return x


def unwrap[T](x: T | None) -> T:
    assert x is not None
    return x
