from dpl import Layer


class Optimizer:
    def __init__(self):
        self.target: Layer | None = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        assert self.target is not None
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data


class GradientClip:
    """勾配クリッピング（max norm）"""

    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def __call__(self, params):
        import numpy as np

        # 全パラメータの勾配のL2ノルムを計算
        total_norm = 0.0
        for param in params:
            if param.grad is not None:
                total_norm += np.sum(param.grad.data ** 2)
        total_norm = np.sqrt(total_norm)

        # クリッピング係数
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in params:
                if param.grad is not None:
                    param.grad.data *= clip_coef
