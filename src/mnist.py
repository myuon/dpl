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
from optimizers.adam import Adam
from computational_graph import get_global_recorder

# データを numpy 配列に変換し、正規化
X_array = X.values if hasattr(X, "values") else X
y_array = y.values if hasattr(y, "values") else y

# 正規化 (0-255 -> 0-1)
X_array = X_array.astype(np.float64) / 255.0

# ラベルを整数に変換
y_array = y_array.astype(np.int64)

# train/test 分割 (最初の60000をtrain、残りをtest)
X_train_flat, X_test_flat = X_array[:60000], X_array[60000:]
y_train, y_test = y_array[:60000], y_array[60000:]

# CNNのために4次元に変形 (N, C, H, W)
X_train = X_train_flat.reshape(-1, 1, 28, 28)
X_test = X_test_flat.reshape(-1, 1, 28, 28)

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
    print(f"Max epochs: {iters // iter_per_epoch}")

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
# CNNの学習（軽量版: Conv -> ReLU -> Pooling -> Affine -> ReLU -> Affine -> Softmax）
import importlib
import simple_cnn

importlib.reload(simple_cnn)
from simple_cnn import SimpleCNN

batch_size = 50
iters = 3000

print("=" * 50)
print("Training SimpleCNN")
print("=" * 50)

network = SimpleCNN(
    input_dim=(1, 28, 28),
    conv_param={"filter_num": 16, "filter_size": 5, "pad": 0, "stride": 1},
    hidden_size=100,
    output_size=10,
)

# 学習前のフィルター重みを保存
initial_filters = network.params["W1"].copy()

optimizer = Adam(lr=0.001)
train_acc_list, test_acc_list, iter_loss_list = train_network(
    network, X_train, y_train, X_test, y_test, optimizer, batch_size, iters
)

# 学習後のフィルター重みを取得
final_filters = network.params["W1"]

# %%
# 学習結果の可視化
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左: Lossの推移（イテレーションごと）
iterations = np.arange(len(iter_loss_list))
axes[0].plot(iterations, iter_loss_list, color="blue", alpha=0.7)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True)

# 右: 精度の推移（エポックごと）
epochs = np.arange(len(train_acc_list))
axes[1].plot(
    epochs, train_acc_list, label="Train Accuracy", color="blue", marker="o", alpha=0.7
)
axes[1].plot(
    epochs, test_acc_list, label="Test Accuracy", color="red", marker="s", alpha=0.7
)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# %%
# フィルター重みの可視化（学習前と学習後）
print("=" * 50)
print("Visualizing Conv Layer Filters")
print("=" * 50)

# フィルターは (filter_num, input_channel, filter_h, filter_w) = (16, 1, 5, 5)
filter_num = initial_filters.shape[0]

# 4x8のグリッドで16個のフィルターを表示（左側4x4: 学習前、右側4x4: 学習後）
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle(
    "Conv Layer Filters: Before Training (left) vs After Training (right)", fontsize=16
)

for i in range(filter_num):
    row = i // 4
    col = i % 4

    # 学習前のフィルター (左側4x4)
    ax_before = axes[row, col]
    filter_before = initial_filters[i, 0]  # (5, 5)
    ax_before.imshow(filter_before, cmap="gray", interpolation="nearest")
    ax_before.set_title(f"Filter {i+1}\nBefore", fontsize=8)
    ax_before.axis("off")

    # 学習後のフィルター (右側4x4)
    ax_after = axes[row, col + 4]
    filter_after = final_filters[i, 0]  # (5, 5)
    ax_after.imshow(filter_after, cmap="gray", interpolation="nearest")
    ax_after.set_title(f"Filter {i+1}\nAfter", fontsize=8)
    ax_after.axis("off")

plt.tight_layout()
plt.show()

# フィルターの統計情報を表示
print(f"\nFilter statistics:")
print(
    f"Before training - min: {initial_filters.min():.4f}, max: {initial_filters.max():.4f}, mean: {initial_filters.mean():.4f}, std: {initial_filters.std():.4f}"
)
print(
    f"After training  - min: {final_filters.min():.4f}, max: {final_filters.max():.4f}, mean: {final_filters.mean():.4f}, std: {final_filters.std():.4f}"
)
print(
    f"Weight change   - L2 norm: {np.linalg.norm(final_filters - initial_filters):.4f}"
)

# %%
# CNNでは活性化値の可視化は省略
# （SimpleCNNにはrecord_activations機能が実装されていません）
print("Skipping activation visualization for CNN model")

# %%
# 計算グラフの可視化
# 小さなバッチで計算グラフを記録
print("=" * 50)
print("Recording computational graph...")
print("=" * 50)

# 計算グラフの記録を有効化
recorder = get_global_recorder()
recorder.enable()

# 小さなバッチで1回だけforward + backwardを実行
x_small_batch = X_train[:5]
y_small_batch = y_train[:5]
grads_for_graph = network.gradient(x_small_batch, y_small_batch)

# 記録を無効化
recorder.disable()

# 画像として生成してmatplotlibで表示（graphvizがインストールされている場合）
try:
    import subprocess
    import tempfile
    import os
    from PIL import Image

    # 一時的にDOTファイルを作成
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as dot_f:
        dot_f.write(recorder.to_dot(direction="TB"))
        dot_file = dot_f.name

    # 一時的なPNGファイルを作成
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as png_f:
        png_file = png_f.name

    try:
        # dotコマンドで画像に変換
        subprocess.run(["dot", "-Tpng", dot_file, "-o", png_file], check=True)

        # 画像を読み込んでmatplotlibで表示
        img = Image.open(png_file)
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Computational Graph (Forward: black, Backward: red)", fontsize=16)
        plt.tight_layout()
        plt.show()

        print("\nComputational graph displayed successfully!")

    finally:
        # 一時ファイルを削除
        os.unlink(dot_file)
        os.unlink(png_file)

except FileNotFoundError:
    print(
        "\nGraphviz is not installed. Please install it to visualize the computational graph:"
    )
    print("  macOS: brew install graphviz")
    print("  Ubuntu/Debian: sudo apt-get install graphviz")
except ImportError:
    print("\nPillow is not installed. Please install it to display the graph:")
    print("  pip install pillow")
except Exception as e:
    print(f"\nFailed to visualize computational graph: {e}")

# 記録をクリア
recorder.clear()
