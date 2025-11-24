import numpy as np
from dpl import Function, Variable, as_variable
from dpl import metal
import jax.numpy as jnp


class ReLU(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
        (x,) = xs
        xp = metal.get_array_module(x)
        y = xp.maximum(0, x)
        return y

    def backward(self, *gys: Variable) -> Variable:
        (x,) = self.inputs
        (gy,) = gys
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x: Variable) -> Variable:
    return ReLU().apply(x)
