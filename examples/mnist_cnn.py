# %%
from dpl import as_variable, DataLoader, no_grad, Trainer
import dpl.functions as F
import dpl.models as M
import dpl.layers as L
import dpl.optimizers as O
import matplotlib.pyplot as plt
import datasets
import dpl
import numpy as np

batch_size = 1000
max_epoch = 15
hidden_size = 1000
lr = 0.001
max_grad = 25.0

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
model = L.Sequential(
    L.Conv2d(30, kernel_size=5, stride=1, pad=0),
    F.relu,
    lambda x: F.pooling(x, kernel_size=2, stride=2),
    lambda x: x.reshape(x.shape[0], -1),
    L.Linear(hidden_size),
    F.relu,
    L.Linear(10),
)
optimizer = O.Adam(lr=lr).setup(model)
# optimizer.add_hook(O.WeightDecay(1e-4))

if dpl.metal.gpu_enable:
    model.to_gpu()
    train_loader.to_gpu()
    test_loader.to_gpu()


# Preprocessing function to reshape images for CNN
def preprocess(x, t):
    # Reshape input to (N, C, H, W) format for CNN
    x = x.reshape(-1, 1, 28, 28)
    return x, t


# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=F.softmax_cross_entropy,
    metric_fn=F.accuracy,
    train_loader=train_loader,
    test_loader=test_loader,
    max_epoch=max_epoch,
    preprocess_fn=preprocess,
    max_grad=max_grad,
    clip_grads=True,
)

x, t = train_set[0]
x = x.reshape(1, 1, 28, 28)
x_var = as_variable(x)

conv_layer = model["l0"]
assert isinstance(conv_layer, L.Conv2d)
conv_layer.prepare(x_var)
conv_weights_initial = conv_layer.W.data_required.copy()

# Train the model
trainer.run()

# %%
# Visualization 1: Loss history over epochs
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, max_epoch + 1), trainer.train_loss_history, label="Train Loss", linewidth=2
)
plt.plot(
    range(1, max_epoch + 1), trainer.test_loss_history, label="Test Loss", linewidth=2
)
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
    range(1, max_epoch + 1),
    trainer.train_metric_history,
    label="Train Accuracy",
    linewidth=2,
)
plt.plot(
    range(1, max_epoch + 1),
    trainer.test_metric_history,
    label="Test Accuracy",
    linewidth=2,
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy Over Epochs (CNN)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Visualization 3: Conv filters
with no_grad():
    # Get the first conv layer weights (first layer in Sequential)
    conv_layer = model["l0"]
    assert isinstance(conv_layer, L.Conv2d)
    conv_weights = conv_layer.W.data_required
    print(f"Conv1 weights shape: {conv_weights.shape}")  # (30, 1, 5, 5)

    # Visualize the first 16 filters (before)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        # Get filter i, channel 0 (since input has 1 channel)
        filter_img = conv_weights_initial[i, 0, :, :]
        im = ax.imshow(filter_img, cmap="gray")
        ax.set_title(f"Filter {i}")
        ax.axis("off")

    plt.suptitle("Learned Conv Filters (5x5) Before Training")
    plt.tight_layout()
    plt.show()

    # Visualize the first 16 filters (after)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        # Get filter i, channel 0 (since input has 1 channel)
        filter_img = conv_weights[i, 0, :, :]
        im = ax.imshow(filter_img, cmap="gray")
        ax.set_title(f"Filter {i}")
        ax.axis("off")

    plt.suptitle("Learned Conv Filters (5x5) After Training")
    plt.tight_layout()
    plt.show()

    # Compute and visualize weight differences
    weight_diff = conv_weights - conv_weights_initial

    # Print difference statistics
    print("\n" + "=" * 50)
    print("Conv Layer Weight Difference Statistics:")
    print("=" * 50)
    print(f"Shape: {weight_diff.shape}")
    print(f"Mean:  {np.mean(weight_diff):.6f}")
    print(f"Std:   {np.std(weight_diff):.6f}")
    print(f"Min:   {np.min(weight_diff):.6f}")
    print(f"Max:   {np.max(weight_diff):.6f}")
    print(f"Abs Mean: {np.mean(np.abs(weight_diff)):.6f}")
    print(f"Abs Max:  {np.max(np.abs(weight_diff)):.6f}")

    # Per-filter statistics
    print("\nPer-filter difference (L2 norm):")
    for i in range(min(16, weight_diff.shape[0])):
        filter_diff = weight_diff[i, 0, :, :]
        l2_norm = np.sqrt(np.sum(filter_diff**2))
        print(
            f"  Filter {i:2d}: L2={l2_norm:.6f}, mean={np.mean(filter_diff):.6f}, std={np.std(filter_diff):.6f}"
        )

    # Visualize the first 16 filter differences
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    vmax = np.max(np.abs(weight_diff[:16]))  # symmetric color scale
    for i in range(16):
        ax = axes[i // 4, i % 4]
        filter_diff = weight_diff[i, 0, :, :]
        ax.imshow(filter_diff, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Filter {i}")
        ax.axis("off")

    plt.suptitle("Conv Filter Weight Differences (After - Before)")
    from matplotlib.colors import Normalize

    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=Normalize(vmin=-vmax, vmax=vmax))
    fig.colorbar(sm, ax=axes, shrink=0.6)
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

        y = model(x_var)
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
