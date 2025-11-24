# %%
import sys
from pathlib import Path

# Add parent directory to path so imports work regardless of where script is run from
sys.path.append(str(Path(__file__).parent.parent))

from dpl import as_variable, DataLoader, no_grad, Variable
import dpl.functions as F
import dpl.models as M
import dpl.layers as L
import dpl.optimizers as O
import matplotlib.pyplot as plt
import datasets
import dpl
import time
import numpy as np


# Define SimpleCNN model
class SimpleCNN(M.Model):
    """Simple CNN for MNIST classification

    Architecture:
    Input (1, 28, 28) -> Conv2d (30 filters, 5x5) -> ReLU -> Pooling (2x2)
    -> Linear (100) -> ReLU -> Linear (10)
    """

    def __init__(self, num_classes: int = 10, hidden_size: int = 100) -> None:
        super().__init__()

        # Conv layer
        self.conv1 = L.Conv2d(30, kernel_size=5, stride=1, pad=0)

        # Fully connected layers
        # After conv1: (28-5+1)/1 = 24, after pool: 24/2 = 12
        # Output size: 30 * 12 * 12 = 4320
        self.fc1 = L.Linear(hidden_size)
        self.fc2 = L.Linear(num_classes)

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs

        # Conv block: Conv -> ReLU -> Pooling
        x = self.conv1.apply(x)
        x = F.relu(x)
        x = F.pooling(x, kernel_size=2, stride=2)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Fully connected layers: Affine -> ReLU -> Affine
        x = self.fc1.apply(x)
        x = F.relu(x)
        x = self.fc2.apply(x)

        return x

    def apply(self, x: Variable) -> Variable:
        out = super().__call__(x)
        assert isinstance(
            out, Variable
        ), f"Output must be a Variable but got {type(out)}"
        return out


batch_size = 1000
max_epoch = 5
hidden_size = 100
lr = 0.1

# Load MNIST dataset
train_set = datasets.MNIST(train=True)
test_set = datasets.MNIST(train=False)

train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Visualize a sample
x, t = train_set[0]
plt.imshow(x.reshape(28, 28), cmap="gray")
plt.title(f"Label: {t}")
plt.show()

print(f"Train set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")
print(f"Sample shape: {x.shape}")

# %%
# Create CNN model and optimizer
model = SimpleCNN(num_classes=10, hidden_size=hidden_size)
optimizer = O.SGD(lr=lr).setup(model)

if dpl.metal.gpu_enable:
    model.to_gpu()
    train_loader.to_gpu()
    test_loader.to_gpu()

# Track loss history
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

# Start timing
start_time = time.time()
epoch_times = []


for epoch in range(max_epoch):
    epoch_start = time.time()
    sum_loss = 0
    sum_accuracy = 0
    batch_count = 0

    for x, t in train_loader:
        # Reshape input to (N, C, H, W) format for CNN
        x = x.reshape(-1, 1, 28, 28)
        x, t = as_variable(x), as_variable(t)

        y = model.apply(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        loss.backward()
        optimizer.update()
        model.cleargrads()

        sum_loss += loss.data_required * len(t)
        sum_accuracy += acc.data_required * len(t)
        batch_count += 1

    train_loss = sum_loss / len(train_set)
    train_loss_history.append(train_loss)
    train_acc_history.append(sum_accuracy / len(train_set))

    # Calculate epoch time and estimate remaining time
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = max_epoch - (epoch + 1)
    estimated_remaining = avg_epoch_time * remaining_epochs

    print(
        f"epoch {epoch+1}/{max_epoch} ({batch_count} batches), "
        f"loss: {train_loss:.4f}, "
        f"accuracy: {sum_accuracy / len(train_set):.4f}, "
        f"time: {epoch_time:.2f}s, "
        f"ETA: {estimated_remaining:.2f}s"
    )

    sum_loss = 0
    sum_accuracy = 0

    with no_grad():
        for x, t in test_loader:
            # Reshape input to (N, C, H, W) format for CNN
            x = x.reshape(-1, 1, 28, 28)
            x, t = as_variable(x), as_variable(t)
            y = model.apply(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            sum_loss += loss.data_required * len(t)
            sum_accuracy += acc.data_required * len(t)

    test_loss = sum_loss / len(test_set)
    test_loss_history.append(test_loss)
    test_acc_history.append(sum_accuracy / len(test_set))

    print(
        f"test loss: {test_loss:.4f}, "
        f"test accuracy: {sum_accuracy / len(test_set):.4f}"
    )


# End timing
end_time = time.time()
total_time = end_time - start_time

print(f"\nTotal training time: {total_time:.2f} seconds")
print(f"Average time per epoch: {total_time / max_epoch:.2f} seconds")

# %%
# Visualization 1: Loss history over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_epoch + 1), train_loss_history, label="Train Loss", linewidth=2)
plt.plot(range(1, max_epoch + 1), test_loss_history, label="Test Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs (CNN)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Visualization 2: Accuracy history over epochs
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, max_epoch + 1), train_acc_history, label="Train Accuracy", linewidth=2
)
plt.plot(range(1, max_epoch + 1), test_acc_history, label="Test Accuracy", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy Over Epochs (CNN)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Visualization 3: Conv filters
with no_grad():
    # Get the first conv layer weights
    conv_weights = model.conv1.W.data_required
    print(f"Conv1 weights shape: {conv_weights.shape}")  # (30, 1, 5, 5)

    # Visualize the first 16 filters
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        # Get filter i, channel 0 (since input has 1 channel)
        filter_img = conv_weights[i, 0, :, :]
        im = ax.imshow(filter_img, cmap="gray")
        ax.set_title(f"Filter {i}")
        ax.axis("off")

    plt.suptitle("Learned Conv Filters (5x5)")
    plt.tight_layout()
    plt.show()

# %%
# Visualization 4: Sample predictions
with no_grad():
    # Get 10 random test samples
    indices = np.random.choice(len(test_set), 10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for idx, test_idx in enumerate(indices):
        x, true_label = test_set[test_idx]
        x_input = x.reshape(1, 1, 28, 28)
        x_var = as_variable(x_input)

        y = model.apply(x_var)
        pred_label = int(np.argmax(y.data_required))

        ax = axes[idx // 5, idx % 5]
        ax.imshow(x.reshape(28, 28), cmap="gray")
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)")
    plt.tight_layout()
    plt.show()

# %%
