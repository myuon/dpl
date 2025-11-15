import numpy as np


class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = x + y
        return out

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = dout * 1
        dy = dout * 1
        return dx, dy
