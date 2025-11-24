import numpy as np
from dpl.core import get_random_module, Variable
from dpl.layers import Layer, Parameter
import dpl.functions as F


class Conv2d(Layer):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride=1,
        pad=0,
        nobias=False,
        in_channels: int | None = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.nobias = nobias
        self.in_channels = in_channels

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_W(np.random)

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels), name="b")

    def _init_W(self, xp):
        assert self.in_channels is not None
        C, OC = self.in_channels, self.out_channels
        KH, KW = (
            self.kernel_size
            if isinstance(self.kernel_size, tuple)
            else (self.kernel_size, self.kernel_size)
        )
        scale = np.sqrt(1.0 / (C * KH * KW))
        W_data = xp.randn(OC, C, KH, KW).astype(np.float32) * scale
        self.W.data = W_data

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def prepare(self, x: Variable) -> None:
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_random_module(x)
            self._init_W(xp)

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs
        self.prepare(x)

        return F.conv2d(x, self.W, self.b, self.stride, self.pad)
