# %%
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from dpl import Model, as_variable
import dpl.layers as L
import dpl.functions as F
import dpl.optimizers as O


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
import datasets
import matplotlib.pyplot as plt

train_set = datasets.SinCurve(train=True)

xs = [p[0] for p in train_set]
ts = [p[1] for p in train_set]
plt.plot(np.arange(len(xs)), xs, label="xs")
plt.plot(np.arange(len(ts)), ts, label="ts")
plt.show()

# %%
max_epoch = 100
hidden_size = 100
bptt_length = 30

model = SimpleRNN(hidden_size=hidden_size, out_size=1)
optimizer = O.Adam().setup(model)

loss_history = []

for epoch in range(max_epoch):
    model.reset_state()
    total_loss = as_variable(np.array(0.0))
    count = 0

    for x, t in train_set:
        y_batch = model(x.reshape(1, 1))
        loss = F.mean_squared_error(y_batch, t.reshape(1, 1))
        count += 1
        total_loss += loss
        if count % bptt_length != 0:
            continue

        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

    avg_loss = total_loss / (len(xs) - 1)
    loss_history.append(avg_loss.data_required.astype(float).item())
    print(f"Epoch {epoch + 1}/{max_epoch}, Loss: {avg_loss}")


# %%
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

# %%
model.reset_state()
predictions = []
true_values = []

for x, t in train_set:
    y = model(x.reshape(1, 1))
    if y.data is not None:
        predictions.append(float(y.data[0, 0]))
        true_values.append(float(t[0]))

plt.figure(figsize=(12, 6))
plt.plot(true_values, label="True Values", alpha=0.7)
plt.plot(predictions, label="Predictions", alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("True Values vs Predictions")
plt.legend()
plt.grid(True)
plt.show()

# %%
