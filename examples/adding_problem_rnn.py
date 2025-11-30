# %%
from dpl import Model, as_variable
import dpl.layers as L
import dpl.functions as F
import dpl.optimizers as O
from dpl.dataloaders import DataLoader
from dpl.trainer import Trainer
import numpy as np


class AddingRNN(Model):
    """
    RNN model for the Adding Problem.

    Input: batch of sequences (N, T, 2) where N is batch size, T is sequence length
    Output: (N, 1) predictions
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = L.TimeRNN(hidden_size)
        self.linear = L.Linear(1, in_size=hidden_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, *xs):
        (x,) = xs  # x: (N, T, 2)

        # TimeRNN が (N, T, 2) -> (N, T, H) を返す前提
        hs = self.rnn(x)  # hs: (N, T, H)

        # 最終時刻の hidden を取り出す
        h_last = hs[:, -1, :]  # (N, H)

        y = self.linear(h_last)  # (N, 1)
        return y


# %%
from datasets.adding_problem import AddingProblem
import matplotlib.pyplot as plt

# Create dataset
sequence_length = 15
num_samples = 10000

train_set = AddingProblem(
    num_samples=num_samples, sequence_length=sequence_length, train=True
)
test_set = AddingProblem(
    num_samples=num_samples, sequence_length=sequence_length, train=False
)

print(f"Dataset created with {len(train_set)} training samples")
print(f"Sequence length: {sequence_length}")
print(f"Input shape: {train_set[0][0].shape}")
print(f"Target shape: {train_set[0][1].shape}")
print(f"Sample target value: {train_set[0][1][0]:.4f}")

# %%
examples = 3
for i in range(examples):
    x, t = train_set[i]
    print(f"Example {i+1}:")
    print(f"Input (first 5 timesteps):\n{x[:]}")
    print(f"Target value: {t[0]:.4f}\n")

# %%
# Training with Trainer and hidden state tracking
max_epoch = 100
hidden_size = 128
batch_size = 32

model = AddingRNN(hidden_size=hidden_size)
optimizer = O.Adam().setup(model)

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Select a fixed sample for tracking hidden states
import dpl

sample_idx = 0
sample_x, sample_t = train_set[sample_idx]
sample_x_batch = sample_x[np.newaxis, :]  # Add batch dimension

# Storage for hidden states across epochs
hidden_states_history = []
epochs_to_record = set(
    list(range(0, max_epoch, max(1, max_epoch // 10))) + [max_epoch - 1]
)  # Record ~10 snapshots + last epoch


# Define callback to record hidden states
def record_hidden_states(trainer):
    """Callback to record hidden states at specific epochs."""
    if trainer.current_epoch in epochs_to_record:
        model.reset_state()
        with dpl.no_grad():
            x_var = as_variable(sample_x_batch)
            # Get hidden states from TimeRNN
            hs = model.rnn(x_var)  # (1, T, H)
            # Extract the hidden states and convert to numpy
            hidden_states = hs.data_required[0]  # (T, H)
            hidden_states_history.append(
                {"epoch": trainer.current_epoch, "states": hidden_states.copy()}
            )


# Create Trainer with callback
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=F.mean_squared_error,
    train_loader=train_loader,
    test_loader=test_loader,
    max_epoch=max_epoch,
    on_epoch_end=record_hidden_states,  # Add callback
)

# Run training
trainer.run()

# Get loss history
loss_history = trainer.train_loss_history
test_loss_history = trainer.test_loss_history

# %%
# Plot training and test loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Train Loss")
plt.plot(test_loss_history, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Test Loss - Adding Problem")
plt.legend()
plt.grid(True)
plt.yscale("log")
plt.show()

# %%
# Visualize hidden states evolution with heatmaps
print(f"\nRecorded hidden states at {len(hidden_states_history)} epochs")
print(f"Sample input shape: {sample_x.shape}")
print(f"Sample target: {sample_t[0]:.4f}")
print(f"Hidden size: {hidden_size}")

# Create a grid of heatmaps showing hidden state evolution
num_snapshots = len(hidden_states_history)
cols = min(3, num_snapshots)
rows = (num_snapshots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
if num_snapshots == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, record in enumerate(hidden_states_history):
    epoch = record["epoch"]
    states = record["states"]  # (T, H)

    ax = axes[idx]
    # Normalize to [0, 1] range
    im = ax.imshow(
        states.T, aspect="auto", cmap="gray", vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_title(f"Epoch {epoch}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Hidden Unit")
    plt.colorbar(im, ax=ax)

# Hide unused subplots
for idx in range(num_snapshots, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.suptitle("Hidden State Evolution During Training", fontsize=16, y=1.01)
plt.show()

# %%
# Plot variance of hidden states across time for each epoch
plt.figure(figsize=(12, 6))

for idx, record in enumerate(hidden_states_history):
    epoch = record["epoch"]
    states = record["states"]  # (T, H)

    # Calculate variance across hidden units at each timestep
    variance_per_timestep = np.var(states, axis=1)

    # Use color gradient based on epoch
    cmap = plt.cm.get_cmap("viridis")
    color = cmap(idx / max(1, len(hidden_states_history) - 1))
    plt.plot(variance_per_timestep, label=f"Epoch {epoch}", color=color, alpha=0.7)

plt.xlabel("Time Step")
plt.ylabel("Variance across Hidden Units")
plt.title("Hidden State Variance Evolution")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Plot mean absolute values of hidden states over time
plt.figure(figsize=(12, 6))

for idx, record in enumerate(hidden_states_history):
    epoch = record["epoch"]
    states = record["states"]  # (T, H)

    # Calculate mean absolute value across hidden units at each timestep
    mean_abs_per_timestep = np.mean(np.abs(states), axis=1)

    cmap = plt.cm.get_cmap("viridis")
    color = cmap(idx / max(1, len(hidden_states_history) - 1))
    plt.plot(mean_abs_per_timestep, label=f"Epoch {epoch}", color=color, alpha=0.7)

plt.xlabel("Time Step")
plt.ylabel("Mean Absolute Hidden State Value")
plt.title("Hidden State Magnitude Evolution")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Evaluation
import dpl

model.reset_state()
predictions = []
true_values = []

with dpl.no_grad():
    for i in range(min(100, len(test_set))):
        model.reset_state()
        x, t = test_set[i]

        x_var = as_variable(x[np.newaxis, :])  # Add batch dimension
        y_pred = model(x_var)  # (1, 1)
        predictions.append(float(y_pred.data_required[0][0]))
        true_values.append(float(t[0]))

# Calculate test MSE
test_mse = np.mean((np.array(predictions) - np.array(true_values)) ** 2)
print(f"Test MSE: {test_mse:.6f}")

# Plot predictions vs true values and residuals
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Predictions vs True Values
ax1 = axes[0]
ax1.scatter(true_values, predictions, alpha=0.5)
ax1.plot([0, 2], [0, 2], "r--", label="Perfect Prediction")
ax1.set_xlabel("True Value")
ax1.set_ylabel("Predicted Value")
ax1.set_title(f"Predictions vs True Values (Test MSE: {test_mse:.6f})")
ax1.legend()
ax1.grid(True)

# Right plot: Residual Distribution
residuals = np.array(predictions) - np.array(true_values)
ax2 = axes[1]
ax2.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
ax2.axvline(x=0, color="r", linestyle="--", label="Zero Residual")
ax2.set_xlabel("Residual (Predicted - True)")
ax2.set_ylabel("Frequency")
ax2.set_title("Residual Distribution")

# Add statistics to the plot
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)
median_residual = np.median(residuals)
ax2.axvline(
    x=mean_residual,
    color="g",
    linestyle="-",
    linewidth=2,
    label=f"Mean: {mean_residual:.4f}",
)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Residual statistics
print("\nResidual Statistics:")
print(f"Mean: {mean_residual:.6f}")
print(f"Std Dev: {std_residual:.6f}")
print(f"Median: {median_residual:.6f}")
print(f"Min: {np.min(residuals):.6f}")
print(f"Max: {np.max(residuals):.6f}")

# %%
# Show some sample predictions
print("\nSample Predictions:")
print("True Value | Predicted | Error")
print("-" * 40)
for i in range(min(10, len(predictions))):
    error = abs(predictions[i] - true_values[i])
    print(f"{true_values[i]:.4f}     | {predictions[i]:.4f}    | {error:.4f}")

# %%
