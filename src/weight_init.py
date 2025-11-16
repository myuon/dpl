import numpy as np
from typing import Callable


def std_weight_init(std: float = 0.01) -> Callable[[int, int], np.ndarray]:
    """標準正規分布による重み初期化

    Args:
        std: 標準偏差

    Returns:
        重み初期化関数
    """

    def init(n_in: int, n_out: int) -> np.ndarray:
        return std * np.random.randn(n_in, n_out)

    return init


def xavier_weight_init() -> Callable[[int, int], np.ndarray]:
    """Xavier初期化（Glorot初期化）

    活性化関数がtanhやsigmoidの場合に有効。
    分散が入力ノード数に依存するように初期化する。

    Returns:
        重み初期化関数
    """

    def init(n_in: int, n_out: int) -> np.ndarray:
        return np.random.randn(n_in, n_out) / np.sqrt(n_in)

    return init


def he_weight_init() -> Callable[[int, int], np.ndarray]:
    """He初期化

    活性化関数がReLUの場合に有効。
    Xavier初期化の2倍の分散を持つ。

    Returns:
        重み初期化関数
    """

    def init(n_in: int, n_out: int) -> np.ndarray:
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

    return init
