import numpy as np
from dpl.core import Variable, Function
from dpl.layers.layer import Parameter, Layer
import dpl.functions as F


class Linear(Layer):
    def __init__(
        self,
        out_size: int,
        no_bias=False,
        dtype=np.float32,
        in_size: int | None = None,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")  # type: ignore
        if self.in_size is not None:
            self._init_W()

        if no_bias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(self.out_size, dtype=self.dtype), name="b")

    def _init_W(self):
        assert self.in_size is not None
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(
            result, Variable
        ), f"Result must be a Variable but got {type(result)}"
        return result

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs

        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        return F.linear(x, self.W, self.b)
