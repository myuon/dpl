# %%
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from dpl import Model, as_variable
import dpl.layers as L
import dpl.functions as F


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.linear = L.Linear(hidden_size, out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, *xs):
        (x,) = xs
        h = self.rnn(x)
        y = self.linear(h)
        return y


# %%
import numpy as np

seq_data = [np.random.randn(1, 1) for _ in range(1000)]
xs = seq_data[0:-1]
ts = seq_data[1:]

model = SimpleRNN(hidden_size=10, out_size=1)

loss, cnt = 0, 0
for x, t in zip(xs, ts):
    x, t = as_variable(x), as_variable(t)
    y = model(x)
    loss += F.mean_squared_error(y, t)
    cnt += 1
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break

print(f"Loss: {loss}")
