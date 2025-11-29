import numpy as np
from dpl import Variable
import dpl.functions as F
import dpl.layers as L


def test_im2col_col2im():
    """im2colとcol2imの動作確認"""
    # 簡単な入力データを作成 (1バッチ, 1チャンネル, 3x3画像)
    x = Variable(np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).astype(np.float32))

    # 2x2のフィルタでim2col
    col = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=0)
    print("im2col result:")
    print(col.data)
    print(f"shape: {col.shape}")  # (4, 4) になるはず (2x2の出力領域 x 2x2のフィルタサイズ)

    # col2imで元に戻す
    x_reconstructed = F.col2im(
        col, input_shape=(1, 1, 3, 3), filter_h=2, filter_w=2, stride=1, pad=0
    )
    print("\ncol2im result:")
    if x_reconstructed.data is not None:
        print(x_reconstructed.data[0, 0])
    print(f"shape: {x_reconstructed.shape}")

    # パディングありの場合
    col_pad = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=1)
    print(f"\nim2col with padding shape: {col_pad.shape}")  # (16, 4) になるはず

    print("✓ test_im2col_col2im passed")


def test_conv2d_layer_forward():
    """Conv2dレイヤーの順伝播のテスト"""
    # 簡単な入力データ (1バッチ, 1チャンネル, 4x4画像)
    x = Variable(np.random.randn(1, 1, 4, 4).astype(np.float32))

    # Conv2dレイヤーの作成 (2フィルタ, kernel_size=3, stride=1, pad=0)
    conv = L.Conv2d(out_channels=2, kernel_size=3, stride=1, pad=0, in_channels=1)

    # 順伝播
    out = conv(x)
    print("Conv2d layer forward output shape:", out.shape)  # (1, 2, 2, 2) になるはず
    print("Output:\n", out.data)

    # 期待される出力形状
    assert out.shape == (
        1,
        2,
        2,
        2,
    ), f"Expected shape (1, 2, 2, 2), got {out.shape}"

    print("✓ test_conv2d_layer_forward passed")


def test_conv2d_layer_backward():
    """Conv2dレイヤーの逆伝播のテスト"""
    # 入力データ
    x = Variable(np.random.randn(2, 1, 4, 4).astype(np.float32))

    # Conv2dレイヤー
    conv = L.Conv2d(out_channels=3, kernel_size=3, stride=1, pad=1, in_channels=1)

    # 順伝播
    out = conv(x)

    # スカラー損失にするためにsum
    loss = F.sum(out)

    # 逆伝播
    loss.backward()

    # 勾配が存在することを確認
    assert x.grad is not None, "x.grad should not be None"
    assert conv.W.grad is not None, "conv.W.grad should not be None"
    assert conv.W.data is not None, "conv.W.data should not be None"
    assert conv.b is not None, "conv.b should not be None"
    assert conv.b.grad is not None, "conv.b.grad should not be None"
    assert conv.b.data is not None, "conv.b.data should not be None"

    print("\nBackward pass check:")
    print(f"x.grad shape: {x.grad.shape}, expected: {x.shape}")
    print(f"conv.W.grad shape: {conv.W.grad.shape}, expected: {conv.W.data.shape}")
    print(f"conv.b.grad shape: {conv.b.grad.shape}, expected: {conv.b.data.shape}")

    # 勾配の形状チェック
    assert x.grad.shape == x.shape, f"x.grad shape mismatch"
    assert (
        conv.W.grad.shape == conv.W.data.shape
    ), f"conv.W.grad shape mismatch"
    assert (
        conv.b.grad.shape == conv.b.data.shape
    ), f"conv.b.grad shape mismatch"

    print("✓ test_conv2d_layer_backward passed")


def test_conv2d_layer_gradient():
    """Conv2dレイヤーの勾配チェック（数値微分との比較）"""
    np.random.seed(42)

    # 小さい入力データでテスト
    x_data = np.random.randn(1, 1, 3, 3).astype(np.float32)
    x = Variable(x_data.copy())

    # Conv2dレイヤー
    conv = L.Conv2d(out_channels=1, kernel_size=2, stride=1, pad=0, in_channels=1)

    # 重みを固定値に設定
    assert conv.W.data is not None
    assert conv.b is not None
    assert conv.b.data is not None
    conv.W.data = np.random.randn(1, 1, 2, 2).astype(np.float32)
    conv.b.data = np.zeros(1).astype(np.float32)

    # 順伝播
    out = conv(x)
    loss = F.sum(out)

    # 逆伝播
    loss.backward()

    # 数値微分でチェック
    assert conv.W.grad is not None
    assert conv.W.grad.data is not None

    W_data_saved = conv.W.data.copy()
    b_data_saved = conv.b.data.copy()

    def f(W_):
        conv_temp = L.Conv2d(out_channels=1, kernel_size=2, stride=1, pad=0, in_channels=1)
        conv_temp.W.data = W_.copy()
        assert conv_temp.b is not None
        conv_temp.b.data = b_data_saved.copy()
        x_temp = Variable(x_data.copy())
        out_temp = conv_temp(x_temp)
        assert out_temp.data is not None
        return float(np.sum(out_temp.data))

    h = 1e-4
    W_test = W_data_saved.copy()
    W_test[0, 0, 0, 0] += h
    grad_numerical = (f(W_test) - f(W_data_saved)) / h
    grad_backprop = float(conv.W.grad.data[0, 0, 0, 0])

    print(f"\nNumerical gradient: {grad_numerical}")
    print(f"Backprop gradient:  {grad_backprop}")
    print(f"Difference: {abs(grad_numerical - grad_backprop)}")

    # 数値微分との差が小さいことを確認
    assert abs(grad_numerical - grad_backprop) < 1e-3, "Gradient check failed"

    print("✓ test_conv2d_layer_gradient passed")


def test_conv2d_multichannel():
    """Conv2dレイヤーの多チャンネル入力のテスト"""
    # 入力データ (2バッチ, 3チャンネル, 8x8画像)
    x = Variable(np.random.randn(2, 3, 8, 8).astype(np.float32))

    # Conv2dレイヤー (4フィルタ, kernel_size=3)
    conv = L.Conv2d(out_channels=4, kernel_size=3, stride=1, pad=0, in_channels=3)

    # 順伝播
    out = conv(x)

    # 期待される出力形状 (2, 4, 6, 6)
    # OH = (8 - 3) / 1 + 1 = 6
    assert out.shape == (2, 4, 6, 6), f"Expected shape (2, 4, 6, 6), got {out.shape}"

    print(f"Multichannel convolution output shape: {out.shape}")
    print("✓ test_conv2d_multichannel passed")


def test_conv2d_with_stride_and_padding():
    """Conv2dレイヤーのstride, padding のテスト"""
    # 入力データ (1バッチ, 1チャンネル, 5x5画像)
    x = Variable(np.random.randn(1, 1, 5, 5).astype(np.float32))

    # stride=2, pad=1
    conv = L.Conv2d(out_channels=2, kernel_size=3, stride=2, pad=1, in_channels=1)

    # 順伝播
    out = conv(x)

    # 期待される出力形状 (1, 2, 3, 3)
    # OH = (5 + 2*1 - 3) / 2 + 1 = 3
    assert out.shape == (1, 2, 3, 3), f"Expected shape (1, 2, 3, 3), got {out.shape}"

    print(f"Conv2d with stride=2, pad=1 output shape: {out.shape}")
    print("✓ test_conv2d_with_stride_and_padding passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Test im2col and col2im")
    print("=" * 60)
    test_im2col_col2im()

    print("\n" + "=" * 60)
    print("Test Conv2d layer forward")
    print("=" * 60)
    test_conv2d_layer_forward()

    print("\n" + "=" * 60)
    print("Test Conv2d layer backward")
    print("=" * 60)
    test_conv2d_layer_backward()

    print("\n" + "=" * 60)
    print("Test Conv2d layer gradient")
    print("=" * 60)
    test_conv2d_layer_gradient()

    print("\n" + "=" * 60)
    print("Test Conv2d multichannel")
    print("=" * 60)
    test_conv2d_multichannel()

    print("\n" + "=" * 60)
    print("Test Conv2d with stride and padding")
    print("=" * 60)
    test_conv2d_with_stride_and_padding()

    print("\n" + "=" * 60)
    print("✓ All convolution tests passed!")
    print("=" * 60)
