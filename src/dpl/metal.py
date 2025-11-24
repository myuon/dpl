import numpy as np
import jax.numpy as jnp


gpu_enable = True


def get_array_module(x):
    from dpl.core import Variable

    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np

    return jnp if isinstance(x, jnp.ndarray) else np


def as_numpy(x: jnp.ndarray | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x

    return np.asarray(x)


def as_jax(x: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    if isinstance(x, jnp.ndarray):
        return x

    return jnp.asarray(x)
