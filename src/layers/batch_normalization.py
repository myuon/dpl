import numpy as np


class BatchNormalization:
    """Batch Normalization層

    ミニバッチごとに入力を正規化し、学習を安定化・高速化する。
    学習時と推論時で異なる動作をする。

    学習時: ミニバッチの平均・分散で正規化
    推論時: 学習時に記録した移動平均で正規化
    """

    def __init__(self, gamma: np.ndarray, beta: np.ndarray, momentum: float = 0.9):
        """
        Args:
            gamma: スケールパラメータ (学習対象)
            beta: シフトパラメータ (学習対象)
            momentum: 移動平均の更新係数
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum

        # 推論時に使用する移動平均
        self.running_mean: np.ndarray | None = None
        self.running_var: np.ndarray | None = None

        # backward時に使用する中間データ
        self.batch_size: int | None = None
        self.xc: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.xn: np.ndarray | None = None

        # 勾配
        self.dgamma: np.ndarray | None = None
        self.dbeta: np.ndarray | None = None

    def forward(self, x: np.ndarray, train_flg: bool = True) -> np.ndarray:
        """順伝播

        Args:
            x: 入力 (batch_size, input_size)
            train_flg: 学習時はTrue、推論時はFalse

        Returns:
            正規化された出力
        """
        if self.running_mean is None:
            # 初回実行時に移動平均を初期化
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            # 学習時: ミニバッチの統計量で正規化
            mu = x.mean(axis=0)  # ミニバッチの平均
            xc = x - mu  # 中心化
            var = np.mean(xc**2, axis=0)  # ミニバッチの分散
            std = np.sqrt(var + 1e-7)  # 標準偏差 (ゼロ除算防止のため小さな値を加算)
            xn = xc / std  # 正規化

            # backward用に保存
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            # 移動平均を更新
            assert self.running_mean is not None
            assert self.running_var is not None
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 推論時: 移動平均で正規化
            assert self.running_mean is not None
            assert self.running_var is not None
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 1e-7)

        # スケール・シフト
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """逆伝播

        Args:
            dout: 出力側から伝わってきた勾配

        Returns:
            入力側への勾配
        """
        assert self.batch_size is not None
        assert self.xc is not None
        assert self.xn is not None
        assert self.std is not None

        # パラメータの勾配
        self.dbeta = dout.sum(axis=0)
        self.dgamma = np.sum(self.xn * dout, axis=0)

        # 正規化の逆伝播
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        return dx
