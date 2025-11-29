# %%
import sys
from pathlib import Path

# Add parent directory to path so imports work regardless of where script is run from
sys.path.append(str(Path(__file__).parent.parent))

from dpl import as_variable, DataLoader, no_grad
import dpl.functions as F
import dpl.models as M
import dpl.optimizers as O
import matplotlib.pyplot as plt
import datasets
import dpl
import time
import numpy as np


# Hyperparameters
batch_size = 300
max_epoch = 100
hidden_size = 300
lr = 1.0

# Load spiral dataset
train_set = datasets.Spiral(train=True)
test_set = datasets.Spiral(train=False)

train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Visualize the dataset
x, t = train_set[0]
print(f"Sample data point: {x}, label: {t}")

plt.figure(figsize=(8, 8))
for i in range(len(train_set)):
    x, t = train_set[i]
    plt.scatter(x[0], x[1], c=["red", "green", "blue"][t], alpha=0.6)
plt.title("Spiral Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.grid(True)
plt.show()

# %%
# Create model and optimizer
model = M.MLP([hidden_size, hidden_size, 3], activation=F.relu)
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

for epoch in range(max_epoch):
    sum_loss = 0
    sum_accuracy = 0

    for x, t in train_loader:
        x, t = as_variable(x), as_variable(t)
        y = model.apply(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        loss.backward()
        optimizer.update()
        model.cleargrads()

        sum_loss += loss.data_required * len(t)
        sum_accuracy += acc.data_required * len(t)

    train_loss = sum_loss / len(train_set)
    train_loss_history.append(train_loss)
    train_acc_history.append(sum_accuracy / len(train_set))

    if (epoch + 1) % 10 == 0:
        print(
            f"epoch {epoch+1}/{max_epoch}, "
            f"loss: {train_loss:.4f}, "
            f"accuracy: {sum_accuracy / len(train_set):.4f}"
        )

    sum_loss = 0
    sum_accuracy = 0

    with no_grad():
        for x, t in test_loader:
            x, t = as_variable(x), as_variable(t)
            y = model.apply(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            sum_loss += loss.data_required * len(t)
            sum_accuracy += acc.data_required * len(t)

    test_loss = sum_loss / len(test_set)
    test_loss_history.append(test_loss)
    test_acc_history.append(sum_accuracy / len(test_set))

# End timing
end_time = time.time()
total_time = end_time - start_time

print(f"\nTotal training time: {total_time:.2f} seconds")
print(f"Average time per epoch: {total_time / max_epoch:.2f} seconds")
print(
    f"Final test accuracy: {test_acc_history[-1]:.4f}, "
    f"Final test loss: {test_loss_history[-1]:.4f}"
)

# %%
# Visualization 1: Loss history over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_epoch + 1), train_loss_history, label="Train Loss", linewidth=2)
plt.plot(range(1, max_epoch + 1), test_loss_history, label="Test Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs")
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
plt.title("Training and Test Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Visualization 3: Decision boundary
h = 0.01  # mesh step size
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict for each point in the mesh
mesh_data = np.c_[xx.ravel(), yy.ravel()]
mesh_data = mesh_data.astype(np.float32)

with no_grad():
    mesh_variable = as_variable(mesh_data)
    Z = model.apply(mesh_variable)
    Z = np.argmax(Z.data_required, axis=1)

Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 10))
plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(4) - 0.5, cmap="RdYlBu")

# Plot training data
for i in range(len(train_set)):
    x, t = train_set[i]
    plt.scatter(
        x[0], x[1], c=["red", "yellow", "blue"][t], edgecolors="black", s=50, alpha=0.8
    )

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Boundary and Training Data")
plt.grid(True)
plt.show()

# %%
