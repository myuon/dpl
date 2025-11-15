import numpy as np


class Sigmoid:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * self.out * (1 - self.out)
        return dx
