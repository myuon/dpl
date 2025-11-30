from dpl.core import Variable
from dpl.layers.layer import StatefulLayer
import dpl.layers as L
import dpl.functions as F


class RNN(StatefulLayer):
    def __init__(self, hidden_size: int, in_size=None):
        super().__init__()
        self.x2h = L.Linear(hidden_size, in_size=in_size)
        self.h2h = L.Linear(hidden_size, in_size=hidden_size, no_bias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, *xs: Variable) -> Variable:
        if self.h is None:
            h_new = F.tanh(self.x2h(xs[0]))
        else:
            h_new = F.tanh(self.x2h(xs[0]) + self.h2h(self.h))
        self.h = h_new
        return h_new
