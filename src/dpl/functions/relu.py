from dpl import UnaryFunction, Variable, ndarray
from dpl.core import metal
import jax.numpy as jnp


class ReLU(UnaryFunction):
    def forward(self, *xs: ndarray) -> ndarray:
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
