import numpy as np
import jax.numpy as jnp
from dpl.core.utils import ndarray


gpu_enable = True


def get_array_module(x):
    from dpl.core import Variable

    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np

    return jnp if isinstance(x, jnp.ndarray) else np


def as_numpy(x: ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, jnp.ndarray):
        return np.asarray(x)

    raise TypeError("Input is not a recognized ndarray type.")


def as_jax(x: ndarray) -> jnp.ndarray:
    if isinstance(x, jnp.ndarray):
        return x
    elif isinstance(x, np.ndarray):
        return jnp.asarray(x)

    raise TypeError("Input is not a recognized ndarray type.")
