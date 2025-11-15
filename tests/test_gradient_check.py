import numpy as np
import pytest
from src.layers.add import AddLayer
from src.layers.mul import MulLayer
from src.layers.relu import Relu
from src.layers.sigmoid import Sigmoid
from src.layers.affine import Affine
from src.layers.softmax_with_loss import SoftmaxWithLoss


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


class TestAddLayer:
    def test_backward(self):
        """AddLayerのbackwardが数値微分と一致することを確認"""
        layer = AddLayer()

        # テストデータ
        x = np.random.randn(3, 4)
        y = np.random.randn(3, 4)
        dout = np.random.randn(3, 4)

        # forward
        out = layer.forward(x, y)

        # backward
        dx, dy = layer.backward(dout)

        # 数値微分によるチェック
        def f_x(x_):
            return np.sum(layer.forward(x_, y) * dout)

        def f_y(y_):
            return np.sum(layer.forward(x, y_) * dout)

        dx_num = numerical_gradient(f_x, x.copy())
        dy_num = numerical_gradient(f_y, y.copy())

        assert np.allclose(dx, dx_num), f"dx mismatch: {dx} vs {dx_num}"
        assert np.allclose(dy, dy_num), f"dy mismatch: {dy} vs {dy_num}"


class TestMulLayer:
    def test_backward(self):
        """MulLayerのbackwardが数値微分と一致することを確認"""
        layer = MulLayer()

        # テストデータ
        x = np.random.randn(3, 4)
        y = np.random.randn(3, 4)
        dout = np.random.randn(3, 4)

        # forward
        out = layer.forward(x, y)

        # backward
        dx, dy = layer.backward(dout)

        # 数値微分によるチェック
        def f_x(x_):
            return np.sum(layer.forward(x_, y) * dout)

        def f_y(y_):
            return np.sum(layer.forward(x, y_) * dout)

        dx_num = numerical_gradient(f_x, x.copy())
        dy_num = numerical_gradient(f_y, y.copy())

        assert np.allclose(dx, dx_num), f"dx mismatch: {dx} vs {dx_num}"
        assert np.allclose(dy, dy_num), f"dy mismatch: {dy} vs {dy_num}"


class TestRelu:
    def test_backward(self):
        """Reluのbackwardが数値微分と一致することを確認"""
        layer = Relu()

        # テストデータ（正負の値を含む）
        x = np.random.randn(5, 6)
        dout = np.random.randn(5, 6)

        # forward
        out = layer.forward(x)

        # backward
        dx = layer.backward(dout.copy())

        # 数値微分によるチェック
        def f(x_):
            layer_temp = Relu()
            return np.sum(layer_temp.forward(x_) * dout)

        dx_num = numerical_gradient(f, x.copy())

        assert np.allclose(dx, dx_num), f"dx mismatch: {dx} vs {dx_num}"


class TestSigmoid:
    def test_backward(self):
        """Sigmoidのbackwardが数値微分と一致することを確認"""
        layer = Sigmoid()

        # テストデータ
        x = np.random.randn(5, 6)
        dout = np.random.randn(5, 6)

        # forward
        out = layer.forward(x)

        # backward
        dx = layer.backward(dout)

        # 数値微分によるチェック
        def f(x_):
            layer_temp = Sigmoid()
            return np.sum(layer_temp.forward(x_) * dout)

        dx_num = numerical_gradient(f, x.copy())

        assert np.allclose(dx, dx_num), f"dx mismatch: {dx} vs {dx_num}"


class TestAffine:
    def test_backward_dx(self):
        """Affineのbackwardでのdxが数値微分と一致することを確認"""
        # テストデータ
        batch_size = 3
        input_size = 4
        output_size = 5

        W = np.random.randn(input_size, output_size)
        b = np.random.randn(output_size)
        x = np.random.randn(batch_size, input_size)
        dout = np.random.randn(batch_size, output_size)

        layer = Affine(W, b)

        # forward
        out = layer.forward(x)

        # backward
        dx = layer.backward(dout)

        # 数値微分によるチェック（入力xに関する勾配）
        def f(x_):
            layer_temp = Affine(W.copy(), b.copy())
            return np.sum(layer_temp.forward(x_) * dout)

        dx_num = numerical_gradient(f, x.copy())

        assert np.allclose(dx, dx_num), f"dx mismatch: {dx} vs {dx_num}"

    def test_backward_dW(self):
        """AffineのbackwardでのdWが数値微分と一致することを確認"""
        # テストデータ
        batch_size = 3
        input_size = 4
        output_size = 5

        W = np.random.randn(input_size, output_size)
        b = np.random.randn(output_size)
        x = np.random.randn(batch_size, input_size)
        dout = np.random.randn(batch_size, output_size)

        layer = Affine(W, b)

        # forward
        out = layer.forward(x)

        # backward
        dx = layer.backward(dout)

        # 数値微分によるチェック（重みWに関する勾配）
        def f(W_):
            layer_temp = Affine(W_, b.copy())
            return np.sum(layer_temp.forward(x.copy()) * dout)

        dW_num = numerical_gradient(f, W.copy())

        assert layer.dW is not None
        assert np.allclose(layer.dW, dW_num), f"dW mismatch: {layer.dW} vs {dW_num}"

    def test_backward_db(self):
        """Affineのbackwardでのdbが数値微分と一致することを確認"""
        # テストデータ
        batch_size = 3
        input_size = 4
        output_size = 5

        W = np.random.randn(input_size, output_size)
        b = np.random.randn(output_size)
        x = np.random.randn(batch_size, input_size)
        dout = np.random.randn(batch_size, output_size)

        layer = Affine(W, b)

        # forward
        out = layer.forward(x)

        # backward
        dx = layer.backward(dout)

        # 数値微分によるチェック（バイアスbに関する勾配）
        def f(b_):
            layer_temp = Affine(W.copy(), b_)
            return np.sum(layer_temp.forward(x.copy()) * dout)

        db_num = numerical_gradient(f, b.copy())

        assert layer.db is not None
        assert np.allclose(layer.db, db_num), f"db mismatch: {layer.db} vs {db_num}"


class TestSoftmaxWithLoss:
    def test_backward(self):
        """SoftmaxWithLossのbackwardが数値微分と一致することを確認"""
        layer = SoftmaxWithLoss()

        # テストデータ
        batch_size = 3
        num_classes = 5
        x = np.random.randn(batch_size, num_classes)
        t = np.random.randint(0, num_classes, batch_size)  # ラベル形式

        # forward
        loss = layer.forward(x, t)

        # backward
        dx = layer.backward()

        # 数値微分によるチェック
        def f(x_):
            layer_temp = SoftmaxWithLoss()
            return layer_temp.forward(x_, t)

        dx_num = numerical_gradient(f, x.copy())

        assert np.allclose(dx, dx_num), f"dx mismatch: {dx} vs {dx_num}"

    def test_backward_onehot(self):
        """SoftmaxWithLossのbackward（one-hot形式）が数値微分と一致することを確認"""
        layer = SoftmaxWithLoss()

        # テストデータ（one-hot形式）
        batch_size = 3
        num_classes = 5
        x = np.random.randn(batch_size, num_classes)
        t = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            t[i, np.random.randint(num_classes)] = 1

        # forward
        loss = layer.forward(x, t)

        # backward
        dx = layer.backward()

        # 数値微分によるチェック
        def f(x_):
            layer_temp = SoftmaxWithLoss()
            return layer_temp.forward(x_, t)

        dx_num = numerical_gradient(f, x.copy())

        assert np.allclose(dx, dx_num), f"dx mismatch: {dx} vs {dx_num}"
