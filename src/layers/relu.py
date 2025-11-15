import numpy as np


class Relu:
    def __init__(self) -> None:
        self.mask: np.ndarray = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx = dout
        return dx
