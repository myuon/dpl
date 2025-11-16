import numpy as np


class Dropout:
    """Dropout層

    学習時にランダムにニューロンを無効化することで過学習を防ぐ。
    推論時は全てのニューロンを使用する。
    """

    def __init__(self, dropout_ratio: float = 0.5) -> None:
        """
        Args:
            dropout_ratio: ドロップアウト率（0.0〜1.0）
        """
        self.dropout_ratio = dropout_ratio
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, train_flg: bool = True) -> np.ndarray:
        """順伝播

        Args:
            x: 入力
            train_flg: 学習時はTrue、推論時はFalse

        Returns:
            出力
        """
        if train_flg:
            # 学習時: ランダムにニューロンを無効化
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 推論時: 全てのニューロンを使用（スケーリングは学習時に行っている）
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """逆伝播

        Args:
            dout: 出力側の勾配

        Returns:
            入力側の勾配
        """
        assert self.mask is not None
        return dout * self.mask
