import numpy as np


class SGD:
    """確率的勾配降下法（Stochastic Gradient Descent）"""

    def __init__(self, lr: float = 0.01) -> None:
        """
        Args:
            lr: 学習率
        """
        self.lr = lr

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """パラメータを更新

        Args:
            params: パラメータの辞書
            grads: 勾配の辞書
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]
