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
import importlib
import n_layer_net
importlib.reload(n_layer_net)
from n_layer_net import NLayerNet
from optimizers.adam import Adam
from weight_init import he_weight_init

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
# 学習関数の定義
def train_network(
    network,
    X_train,
    y_train,
    X_test,
    y_test,
    optimizer,
    batch_size=100,
    iters=10000,
):
    """ニューラルネットワークを学習する

    Args:
        network: 学習対象のネットワーク
        X_train: 訓練データ
        y_train: 訓練ラベル
        X_test: テストデータ
        y_test: テストラベル
        optimizer: オプティマイザー（SGD, Adamなど）
        batch_size: ミニバッチサイズ
        iters: イテレーション数

    Returns:
        tuple: (train_acc_list, test_acc_list, iter_loss_list)
    """

    # 学習経過を記録
    train_acc_list = []
    test_acc_list = []
    iter_loss_list = []  # イテレーションごとのloss

    # 1エポックあたりのイテレーション数
    iter_per_epoch = max(X_train.shape[0] // batch_size, 1)

    # 学習を実行
    print(f"Starting training...")
    print(f"Batch size: {batch_size}, Iterations: {iters}")
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

        # イテレーションごとにミニバッチのlossを記録
        loss = network.loss(x_batch, y_batch)
        iter_loss_list.append(loss)

        # エポックごとに精度を計算
        if i % iter_per_epoch == 0:
            epoch = i // iter_per_epoch
            train_acc = network.accuracy(X_train, y_train)
            test_acc = network.accuracy(X_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch}: train loss = {loss:.4f}, train acc = {train_acc:.4f}, test acc = {test_acc:.4f}, time = {elapsed_time:.2f}s"
            )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")

    return train_acc_list, test_acc_list, iter_loss_list


# %%
# Batch Normalizationの比較実験
batch_size = 100
iters = 10000

# 1. Batch Normalizationなし
print("=" * 50)
print("Training WITHOUT Batch Normalization")
print("=" * 50)
network_without_bn = NLayerNet(
    input_size=784,
    hidden_size=50,
    output_size=10,
    hidden_layer_num=4,
    weight_initializer=he_weight_init(),
    use_batchnorm=False,
)
optimizer_without_bn = Adam(lr=0.001)
train_acc_without_bn, test_acc_without_bn, iter_loss_without_bn = train_network(
    network_without_bn, X_train, y_train, X_test, y_test, optimizer_without_bn, batch_size, iters
)

# 2. Batch Normalizationあり
print("\n" + "=" * 50)
print("Training WITH Batch Normalization")
print("=" * 50)
network_with_bn = NLayerNet(
    input_size=784,
    hidden_size=50,
    output_size=10,
    hidden_layer_num=4,
    weight_initializer=he_weight_init(),
    use_batchnorm=True,
)
optimizer_with_bn = Adam(lr=0.001)
train_acc_with_bn, test_acc_with_bn, iter_loss_with_bn = train_network(
    network_with_bn, X_train, y_train, X_test, y_test, optimizer_with_bn, batch_size, iters
)

# 最後に学習したネットワークを保持（活性化値の可視化用）
network = network_with_bn
train_acc_list = train_acc_with_bn
test_acc_list = test_acc_with_bn
iter_loss_list = iter_loss_with_bn

# %%
# Batch Normalizationの比較可視化
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左: Lossの推移（イテレーションごと）
iterations = np.arange(len(iter_loss_without_bn))
axes[0].plot(iterations, iter_loss_without_bn, label="Without BatchNorm", color="blue", alpha=0.7)
axes[0].plot(iterations, iter_loss_with_bn, label="With BatchNorm", color="red", alpha=0.7)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss Comparison")
axes[0].legend()
axes[0].grid(True)

# 右: テスト精度の推移（エポックごと）
epochs = np.arange(len(test_acc_without_bn))
axes[1].plot(epochs, test_acc_without_bn, label="Without BatchNorm", color="blue", marker="o", alpha=0.7)
axes[1].plot(epochs, test_acc_with_bn, label="With BatchNorm", color="red", marker="s", alpha=0.7)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Test Accuracy Comparison")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# %%
# 各層の活性化値のヒストグラムを可視化
# テストデータの一部を使って活性化値を記録
sample_size = 1000
x_sample = X_test[:sample_size]

# 活性化値を記録しながら予測
network.predict(x_sample, record_activations=True)

# ReLU層の活性化値のみを抽出
relu_activations = {
    name: activations
    for name, activations in network.activations.items()
    if "Relu" in name
}

# ヒストグラムを描画
num_layers = len(relu_activations)
fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))

# 1層の場合はaxesをリストに変換
if num_layers == 1:
    axes = [axes]

for idx, (layer_name, activations) in enumerate(relu_activations.items()):
    ax = axes[idx]
    ax.hist(activations.flatten(), bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{layer_name} Activation Distribution")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
