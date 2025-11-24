import numpy as np
import jax.numpy as jnp
from typing import TypeAlias


ndarray: TypeAlias = np.ndarray | jnp.ndarray
ndarray_types = (np.ndarray, jnp.ndarray)


def as_nparray(x, array_module=np) -> np.ndarray:
    if np.isscalar(x):
        return array_module.array(x)

    return x


def unwrap[T](x: T | None) -> T:
    assert x is not None
    return x
