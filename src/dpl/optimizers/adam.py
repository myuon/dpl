import numpy as np
from dpl.optimizers.optimizer import Optimizer
from dpl import Variable


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict[int, np.ndarray] = {}  # 1次モーメント
        self.v: dict[int, np.ndarray] = {}  # 2次モーメント
        self.t: dict[int, int] = {}  # パラメータごとのタイムステップ

    def update_one(self, param: Variable) -> None:
        assert param.grad is not None

        param_id = id(param)

        # 初回の場合、モーメントとタイムステップを初期化
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param.data)
            self.v[param_id] = np.zeros_like(param.data)
            self.t[param_id] = 0

        self.t[param_id] += 1

        # 1次モーメント (momentum) の更新
        self.m[param_id] = (
            self.beta1 * self.m[param_id] + (1 - self.beta1) * param.grad.data_required
        )

        # 2次モーメント (RMSprop) の更新
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (
            param.grad.data_required**2
        )

        # バイアス補正
        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t[param_id])
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t[param_id])

        # パラメータの更新
        param.data = param.data_required - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
