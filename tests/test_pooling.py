import sys
from pathlib import Path

# srcディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from layers.pooling import Pooling


def test_pooling_forward():
    """プーリング層の順伝播のテスト"""
    # 簡単な入力データ (1バッチ, 1チャンネル, 4x4画像)
    x = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=float)

    print("Input:")
    print(x[0, 0])

    # 2x2のMax Pooling (stride=2)
    pool = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
    out = pool.forward(x)

    print("\nPooling output (2x2, stride=2):")
    print(out[0, 0])
    print(f"Output shape: {out.shape}")  # (1, 1, 2, 2) になるはず

    # 期待される出力: [[6, 8], [14, 16]]
    expected = np.array([[[[6, 8],
                           [14, 16]]]], dtype=float)

    if np.allclose(out, expected):
        print("✓ Forward pass is correct!")
    else:
        print("✗ Forward pass is incorrect!")
        print("Expected:")
        print(expected[0, 0])


def test_pooling_backward():
    """プーリング層の逆伝播のテスト"""
    # 入力データ
    x = np.random.randn(2, 3, 4, 4)  # 2バッチ, 3チャンネル, 4x4

    # プーリング層
    pool = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)

    # 順伝播
    out = pool.forward(x)
    print(f"\nPooling output shape: {out.shape}")  # (2, 3, 2, 2) になるはず

    # 適当な勾配を流す
    dout = np.random.randn(*out.shape)

    # 逆伝播
    dx = pool.backward(dout)

    print(f"dx shape: {dx.shape}, expected: {x.shape}")

    if dx.shape == x.shape:
        print("✓ Backward shape is correct!")
    else:
        print("✗ Backward shape is incorrect!")

    # 勾配チェック（簡易版）
    # プーリングの性質：最大値の位置にのみ勾配が流れる
    print("\nGradient flow check:")

    # 簡単な例で確認
    x_simple = np.array([[[[1, 2],
                           [3, 4]]]], dtype=float)

    pool_simple = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
    out_simple = pool_simple.forward(x_simple)

    # 勾配を1だけ流す
    dout_simple = np.ones_like(out_simple)
    dx_simple = pool_simple.backward(dout_simple)

    print("Input:")
    print(x_simple[0, 0])
    print("Gradient:")
    print(dx_simple[0, 0])
    print("(Gradient should be 1 only at the max position [1,1])")

    # [1,1]の位置（値が4）にのみ勾配が流れるはず
    expected_grad = np.array([[[[0, 0],
                                [0, 1]]]], dtype=float)

    if np.allclose(dx_simple, expected_grad):
        print("✓ Gradient flow is correct!")
    else:
        print("✗ Gradient flow is incorrect!")


def test_pooling_with_stride():
    """異なるストライドでのテスト"""
    x = np.random.randn(1, 1, 8, 8)

    print("\n" + "=" * 50)
    print("Test different stride settings")
    print("=" * 50)

    for stride in [1, 2, 3]:
        pool = Pooling(pool_h=2, pool_w=2, stride=stride, pad=0)
        out = pool.forward(x)
        print(f"Stride={stride}: Input {x.shape} -> Output {out.shape}")

        # 逆伝播も確認
        dout = np.ones_like(out)
        dx = pool.backward(dout)
        print(f"  Backward: {dx.shape}")


if __name__ == "__main__":
    print("=" * 50)
    print("Test Pooling forward")
    print("=" * 50)
    test_pooling_forward()

    print("\n" + "=" * 50)
    print("Test Pooling backward")
    print("=" * 50)
    test_pooling_backward()

    print("\n")
    test_pooling_with_stride()
