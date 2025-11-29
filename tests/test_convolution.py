import numpy as np
from layers.convolution import Convolution, im2col, col2im


def test_im2col_col2im():
    """im2colとcol2imの動作確認"""
    # 簡単な入力データを作成 (1バッチ, 1チャンネル, 3x3画像)
    x = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=float)

    # 2x2のフィルタでim2col
    col = im2col(x, filter_h=2, filter_w=2, stride=1, pad=0)
    print("im2col result:")
    print(col)
    print(
        f"shape: {col.shape}"
    )  # (4, 4) になるはず (2x2の出力領域 x 2x2のフィルタサイズ)

    # col2imで元に戻す
    x_reconstructed = col2im(col, x.shape, filter_h=2, filter_w=2, stride=1, pad=0)
    print("\ncol2im result:")
    print(x_reconstructed[0, 0])
    print(f"shape: {x_reconstructed.shape}")

    # パディングありの場合
    col_pad = im2col(x, filter_h=2, filter_w=2, stride=1, pad=1)
    print(f"\nim2col with padding shape: {col_pad.shape}")  # (16, 4) になるはず


def test_convolution_forward():
    """畳み込み層の順伝播のテスト"""
    # 簡単な入力データ (1バッチ, 1チャンネル, 4x4画像)
    x = np.random.randn(1, 1, 4, 4)

    # フィルタの重み (2フィルタ, 1チャンネル, 3x3)
    W = np.random.randn(2, 1, 3, 3)
    b = np.random.randn(2)

    # 畳み込み層の作成
    conv = Convolution(W, b, stride=1, pad=0)

    # 順伝播
    out = conv.forward(x)
    print("Convolution forward output shape:", out.shape)  # (1, 2, 2, 2) になるはず
    print("Output:\n", out)


def test_convolution_gradient():
    """畳み込み層の勾配チェック（数値微分との比較）"""
    # 入力データ
    x = np.random.randn(2, 1, 4, 4)  # 2バッチ, 1チャンネル, 4x4

    # フィルタ
    W = np.random.randn(3, 1, 3, 3)  # 3フィルタ, 1チャンネル, 3x3
    b = np.random.randn(3)

    conv = Convolution(W, b, stride=1, pad=1)

    # 順伝播
    out = conv.forward(x)

    # 適当な勾配を流す
    dout = np.random.randn(*out.shape)

    # 逆伝播
    dx = conv.backward(dout)

    print("\nGradient check:")
    print(f"dx shape: {dx.shape}, expected: {x.shape}")
    print(
        f"dW shape: {conv.dW.shape if conv.dW is not None else None}, expected: {W.shape}"
    )
    print(
        f"db shape: {conv.db.shape if conv.db is not None else None}, expected: {b.shape}"
    )

    # 数値微分でチェック（簡易版）
    def f(W_):
        conv_temp = Convolution(W_, b, stride=1, pad=1)
        return np.sum(conv_temp.forward(x))

    h = 1e-4
    W_test = W.copy()
    W_test[0, 0, 0, 0] += h
    grad_numerical = (f(W_test) - f(W)) / h

    conv.forward(x)
    dout_ones = np.ones_like(out)
    conv.backward(dout_ones)

    grad_backprop = conv.dW[0, 0, 0, 0] if conv.dW is not None else 0

    print(f"\nNumerical gradient: {grad_numerical}")
    print(f"Backprop gradient:  {grad_backprop}")
    print(f"Difference: {abs(grad_numerical - grad_backprop)}")

    if abs(grad_numerical - grad_backprop) < 1e-5:
        print("✓ Gradient check passed!")
    else:
        print("✗ Gradient check failed!")


if __name__ == "__main__":
    print("=" * 50)
    print("Test im2col and col2im")
    print("=" * 50)
    test_im2col_col2im()

    print("\n" + "=" * 50)
    print("Test Convolution forward")
    print("=" * 50)
    test_convolution_forward()

    print("\n" + "=" * 50)
    print("Test Convolution gradient")
    print("=" * 50)
    test_convolution_gradient()
