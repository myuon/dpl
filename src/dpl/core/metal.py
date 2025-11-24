import numpy as np
import jax.numpy as jnp
import jax
from dpl.core.utils import ndarray


gpu_enable = True


def get_array_module(x):
    from dpl.core import Variable

    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np

    return jnp if isinstance(x, jnp.ndarray) else np


class RandomGen:
    def __init__(self, seed=0):
        self.key = jax.random.PRNGKey(seed)

    def randn(self, *shape, dtype=jnp.float32):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.normal(subkey, shape, dtype=dtype)


rgen = RandomGen(42)


def get_random_module(x):
    from dpl.core import Variable

    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np.random

    return rgen if isinstance(x, jnp.ndarray) else np.random


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
