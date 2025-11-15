import numpy as np


class MulLayer:
    def __init__(self) -> None:
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
