from dpl import Variable
from dpl.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param: Variable) -> None:
        param.data -= self.lr * param.grad_required.data
