import numpy as np
import pytest
import sys
from pathlib import Path

# src/dpl をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "dpl"))

from function import Function
from variable import Variable
from utils import as_nparray


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


def numerical_gradient(f, x, h=1e-4):
    """
    数値微分による勾配計算

    Args:
        f: スカラー値を返す関数
        x: 入力配列
        h: 微小な値

    Returns:
        勾配の配列
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])  # type: ignore

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 勾配の計算
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 値を元に戻す
        x[idx] = tmp_val
        it.iternext()

    return grad


class TestSquare:
    def test_forward(self):
        """Squareのforwardが正しく計算されることを確認"""
        square = Square()
        x = Variable(np.array(2.0))
        y = square(x)
        assert y.data == 4.0

        # 配列の場合
        x = Variable(np.array([1.0, 2.0, 3.0]))
        y = square(x)
        expected = np.array([1.0, 4.0, 9.0])
        assert np.allclose(y.data, expected)

    def test_backward(self):
        """Squareのbackwardが数値微分と一致することを確認"""
        square = Square()

        # スカラー値のテスト
        x_data = np.array(3.0)
        x = Variable(x_data)
        y = square(x)

        # 出力に対する勾配を設定
        dout = np.array(1.0)

        # backward
        gx = square.backward(dout)

        # 数値微分によるチェック
        def f(x_):
            return np.sum(x_ ** 2 * dout)

        gx_num = numerical_gradient(f, x_data.copy())

        assert np.allclose(gx, gx_num), f"gx mismatch: {gx} vs {gx_num}"

    def test_backward_array(self):
        """Squareのbackward（配列）が数値微分と一致することを確認"""
        square = Square()

        # 配列のテスト
        x_data = np.random.randn(3, 4)
        x = Variable(x_data)
        y = square(x)

        # 出力に対する勾配を設定
        dout = np.random.randn(3, 4)

        # backward
        gx = square.backward(dout)

        # 数値微分によるチェック
        def f(x_):
            return np.sum(x_ ** 2 * dout)

        gx_num = numerical_gradient(f, x_data.copy())

        assert np.allclose(gx, gx_num), f"gx mismatch: {gx} vs {gx_num}"

    def test_backward_with_variable(self):
        """Variableのbackwardメソッドを使った勾配計算のテスト"""
        x = Variable(np.array(3.0))
        square = Square()
        y = square(x)
        y.backward()

        # dy/dx = 2x = 2 * 3 = 6
        expected_grad = 6.0
        assert np.allclose(x.grad, expected_grad), f"gradient mismatch: {x.grad} vs {expected_grad}"

    def test_backward_chain(self):
        """連鎖律を使った勾配計算のテスト (x^2)^2 = x^4"""
        x = Variable(np.array(2.0))
        square1 = Square()
        square2 = Square()

        y = square1(x)  # y = x^2
        z = square2(y)  # z = y^2 = (x^2)^2 = x^4

        z.backward()

        # dz/dx = 4x^3 = 4 * 2^3 = 32
        expected_grad = 32.0
        assert np.allclose(x.grad, expected_grad), f"gradient mismatch: {x.grad} vs {expected_grad}"
