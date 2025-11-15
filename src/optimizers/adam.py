import numpy as np


class Adam:
    """Adam (Adaptive Moment Estimation)

    Adam は、各パラメータの適応的な学習率を計算する最適化アルゴリズム。
    勾配の1次モーメント（平均）と2次モーメント（非中心分散）の推定値を使用する。
    """

    def __init__(
        self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ) -> None:
        """
        Args:
            lr: 学習率
            beta1: 1次モーメント推定の減衰率
            beta2: 2次モーメント推定の減衰率
            eps: ゼロ除算を防ぐための小さな値
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict[str, np.ndarray] = {}  # 1次モーメント
        self.v: dict[str, np.ndarray] = {}  # 2次モーメント
        self.t = 0  # タイムステップ

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """パラメータを更新

        Args:
            params: パラメータの辞書
            grads: 勾配の辞書
        """
        # 初回のみ、モーメントを初期化
        if not self.m:
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.t += 1

        for key in params.keys():
            # 1次モーメント (momentum) の更新
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # 2次モーメント (RMSprop) の更新
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # バイアス補正
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # パラメータの更新
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
