# %%
import sys

sys.path.append("..")

from dpl import as_variable, DataLoader, no_grad
import dpl.functions as F
import dpl.models as M
import dpl.optimizers as O
import matplotlib.pyplot as plt
import datasets
import dpl
import time


batch_size = 500
max_epoch = 5
hidden_size = 1000
lr = 0.1


train_set = datasets.MNIST(train=True)
test_set = datasets.MNIST(train=False)

train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

for i in range(6):
    x, t = train_set[i]
    plt.subplot(2, 3, i + 1)
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.title(f"Label: {t}")
    plt.axis("off")
plt.show()

# %%
# Create model and optimizer
model = M.MLP(
    [hidden_size, hidden_size, hidden_size, hidden_size, 10], activation=F.relu
)
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
    batch_count = 0

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
        batch_count += 1

    train_loss = sum_loss / len(train_set)
    train_loss_history.append(train_loss)
    train_acc_history.append(sum_accuracy / len(train_set))

    print(
        f"epoch {epoch+1}/{max_epoch} ({batch_count} batches), "
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
plt.title("Training and Test Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# %%
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
