import sys
from pathlib import Path

# srcディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from simple_cnn import SimpleCNN


def test_cnn_construction():
    """CNNモデルの構築テスト"""
    print("Testing CNN construction...")

    # MNISTサイズの入力
    cnn = SimpleCNN(
        input_dim=(1, 28, 28),
        conv_param={
            'filter_num': 16,
            'filter_size': 5,
            'pad': 0,
            'stride': 1
        },
        hidden_size=100,
        output_size=10
    )

    print(f"✓ CNN model created successfully")
    print(f"  Number of layers: {len(cnn.layers)}")
    print(f"  Layer names: {list(cnn.layers.keys())}")


def test_cnn_forward():
    """CNNの順伝播テスト"""
    print("\nTesting CNN forward pass...")

    cnn = SimpleCNN(
        input_dim=(1, 28, 28),
        conv_param={
            'filter_num': 16,
            'filter_size': 5,
            'pad': 0,
            'stride': 1
        },
        hidden_size=100,
        output_size=10
    )

    # ダミー入力データ (2バッチ, 1チャンネル, 28x28)
    x = np.random.randn(2, 1, 28, 28)

    # 順伝播（推論モード）
    y = cnn.predict(x, train_flg=False)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Expected output shape: (2, 10)")

    assert y.shape == (2, 10), f"Output shape mismatch: {y.shape} != (2, 10)"
    print("✓ Forward pass successful")


def test_cnn_backward():
    """CNNの逆伝播テスト"""
    print("\nTesting CNN backward pass...")

    cnn = SimpleCNN(
        input_dim=(1, 28, 28),
        conv_param={
            'filter_num': 16,
            'filter_size': 5,
            'pad': 0,
            'stride': 1
        },
        hidden_size=50,
        output_size=10
    )

    # ダミーデータ
    x = np.random.randn(2, 1, 28, 28)
    t = np.array([3, 7])  # ラベル

    # 勾配計算
    grads = cnn.gradient(x, t)

    print(f"  Number of gradients: {len(grads)}")
    print(f"  Gradient keys: {list(grads.keys())}")

    # 全ての勾配が存在し、形状が正しいことを確認
    for key, grad in grads.items():
        param = cnn.params[key]
        assert grad.shape == param.shape, f"Gradient shape mismatch for {key}: {grad.shape} != {param.shape}"
        print(f"  {key}: {grad.shape} ✓")

    print("✓ Backward pass successful")


def test_cnn_loss_and_accuracy():
    """損失と精度の計算テスト"""
    print("\nTesting loss and accuracy calculation...")

    cnn = SimpleCNN(
        input_dim=(1, 28, 28),
        conv_param={
            'filter_num': 16,
            'filter_size': 5,
            'pad': 0,
            'stride': 1
        },
        hidden_size=50,
        output_size=10
    )

    # ダミーデータ
    x = np.random.randn(10, 1, 28, 28)
    t = np.random.randint(0, 10, size=10)

    # 損失計算
    loss = cnn.loss(x, t)
    print(f"  Loss: {loss}")
    assert loss > 0, "Loss should be positive"

    # 精度計算
    acc = cnn.accuracy(x, t)
    print(f"  Accuracy: {acc}")
    assert 0 <= acc <= 1, f"Accuracy should be in [0, 1], got {acc}"

    print("✓ Loss and accuracy calculation successful")


def test_different_input_sizes():
    """異なる入力サイズでのテスト"""
    print("\nTesting different input sizes...")

    # 32x32の入力
    cnn = SimpleCNN(
        input_dim=(3, 32, 32),  # RGB画像
        conv_param={
            'filter_num': 16,
            'filter_size': 5,
            'pad': 0,
            'stride': 1
        },
        hidden_size=100,
        output_size=10
    )

    x = np.random.randn(2, 3, 32, 32)
    y = cnn.predict(x, train_flg=False)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (2, 10), f"Output shape mismatch: {y.shape} != (2, 10)"
    print("✓ Different input size test successful")


if __name__ == "__main__":
    print("=" * 50)
    print("SimpleCNN Tests")
    print("=" * 50)

    test_cnn_construction()
    test_cnn_forward()
    test_cnn_backward()
    test_cnn_loss_and_accuracy()
    test_different_input_sizes()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
