# %%
from dpl import Model, as_variable
import dpl.layers as L
import dpl.functions as F
import dpl.optimizers as O
from dpl.dataloaders import SequentialDataLoader


class SimpleLSTM(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.lstm = L.LSTM(hidden_size)
        self.linear = L.Linear(out_size, in_size=hidden_size)

    def reset_state(self):
        self.lstm.reset_state()

    def forward(self, *xs):
        (x,) = xs
        h = self.lstm(x)
        y = self.linear(h)
        return y


# %%
import numpy as np
import datasets
import matplotlib.pyplot as plt

train_set = datasets.SinCurve(train=True)
test_set = datasets.SinCurve(train=False)

xs = [p[0] for p in train_set]
ts = [p[1] for p in train_set]
plt.plot(np.arange(len(xs)), xs, label="xs")
plt.plot(np.arange(len(ts)), ts, label="ts")
plt.legend()
plt.title("Training Data: Sin Curve")
plt.show()

# %%
max_epoch = 100
hidden_size = 100
bptt_length = 30
batch_size = 30

model = SimpleLSTM(hidden_size=hidden_size, out_size=1)
optimizer = O.Adam().setup(model)

# Create SequentialDataLoader
dataloader = SequentialDataLoader(
    train_set, batch_size=batch_size, bptt_length=bptt_length
)

loss_history = []

for epoch in range(max_epoch):
    model.reset_state()
    total_loss = as_variable(np.array(0.0))
    loss_count = 0

    for xs_batch, ts_batch in dataloader:
        # xs_batch shape: (batch_size, bptt_length)
        # ts_batch shape: (batch_size, bptt_length)

        # Process each time step in the sequence
        loss = as_variable(np.array(0.0))
        for t in range(xs_batch.shape[1]):
            x = as_variable(xs_batch[:, t].reshape(batch_size, 1))
            target = as_variable(ts_batch[:, t].reshape(batch_size, 1))
            y = model(x)
            loss += F.mean_squared_error(y, target)

        # Average loss over the sequence
        loss = loss / xs_batch.shape[1]

        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

        total_loss += loss
        loss_count += 1

    avg_loss = total_loss / loss_count
    loss_history.append(avg_loss.data_required.astype(float).item())
    print(f"Epoch {epoch + 1}/{max_epoch}, Loss: {avg_loss}")


# %%
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training Loss")
plt.grid(True)
plt.show()

# %%
import dpl

model.reset_state()
predictions = []
true_values = []

with dpl.no_grad():
    for x, t in test_set:
        y = model(x.reshape(1, 1))
        if y.data is not None:
            predictions.append(float(y.data[0, 0]))
            true_values.append(float(t[0]))

plt.figure(figsize=(12, 6))
plt.plot(true_values, label="True Values", alpha=0.7)
plt.plot(predictions, label="Predictions", alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("LSTM: True Values vs Predictions")
plt.legend()
plt.grid(True)
plt.show()

# %%
