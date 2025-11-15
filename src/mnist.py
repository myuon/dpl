# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# %%
# MNISTデータセットの読み込み
print("MNISTデータセットを読み込んでいます...")
mnist = fetch_openml("mnist_784", version=1, parser="auto")
X, y = mnist["data"], mnist["target"]

print(f"データセット読み込み完了")
print(f"特徴量の形状: {X.shape}")
print(f"ラベルの形状: {y.shape}")

# %%
# データの確認
print(f"総サンプル数: {len(X)}")
print(f"各サンプルの特徴量数: {X.shape[1]}")
print(f"画像サイズ: 28x28 = {28*28}")
print(f"\nラベルの種類: {np.unique(y)}")
print(f"各クラスのサンプル数:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  数字 {label}: {count}個")

# %%
# いくつかの画像を表示
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("MNIST Sample Images", fontsize=16)

for i, ax in enumerate(axes.flat):
    # 画像データを28x28に変形
    image = (
        X.iloc[i].values.reshape(28, 28) if hasattr(X, "iloc") else X[i].reshape(28, 28)
    )
    label = y.iloc[i] if hasattr(y, "iloc") else y[i]

    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
# データの前処理とトレーニング準備
import time
from two_layer_net import TwoLayerNet
from optimizers.adam import Adam

# データを numpy 配列に変換し、正規化
X_array = X.values if hasattr(X, "values") else X
y_array = y.values if hasattr(y, "values") else y

# 正規化 (0-255 -> 0-1)
X_array = X_array.astype(np.float64) / 255.0

# ラベルを整数に変換
y_array = y_array.astype(np.int64)

# train/test 分割 (最初の60000をtrain、残りをtest)
X_train, X_test = X_array[:60000], X_array[60000:]
y_train, y_test = y_array[:60000], y_array[60000:]

print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# %%
# ニューラルネットワークの学習
batch_size = 100
iters = 10000
learning_rate = 0.001

# ネットワークの初期化
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# オプティマイザーの初期化
optimizer = Adam(lr=learning_rate)

# 学習経過を記録
train_acc_list = []
test_acc_list = []
train_loss_list = []

# 1エポックあたりのイテレーション数
iter_per_epoch = max(X_train.shape[0] // batch_size, 1)

# 学習を実行
print(f"Starting training...")
print(f"Batch size: {batch_size}, Iterations: {iters}, Learning rate: {learning_rate}")
print(f"Iterations per epoch: {iter_per_epoch}")

start_time = time.time()

for i in range(iters):
    # ミニバッチの取得
    batch_mask = np.random.choice(X_train.shape[0], batch_size)
    x_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, y_batch)

    # パラメータの更新
    optimizer.update(network.params, grad)

    # エポックごとに精度とlossを計算
    if i % iter_per_epoch == 0:
        epoch = i // iter_per_epoch
        train_loss = network.loss(X_train, y_train)
        train_acc = network.accuracy(X_train, y_train)
        test_acc = network.accuracy(X_test, y_test)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch}: train loss = {train_loss:.4f}, train acc = {train_acc:.4f}, test acc = {test_acc:.4f}, time = {elapsed_time:.2f}s"
        )

end_time = time.time()
total_time = end_time - start_time
print(f"\nTraining completed in {total_time:.2f} seconds")

# %%
# 精度とlossの推移をグラフ化
epochs = np.arange(len(train_acc_list))

# 2つのサブプロットを作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 精度のグラフ
ax1.plot(epochs, train_acc_list, label="Train accuracy", marker="o")
ax1.plot(epochs, test_acc_list, label="Test accuracy", marker="s")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_title("Accuracy Progress")
ax1.legend()
ax1.grid(True)

# Lossのグラフ
ax2.plot(epochs, train_loss_list, label="Train loss", marker="o", color="red")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_title("Loss Progress")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
