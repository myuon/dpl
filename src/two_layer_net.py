import numpy as np
from collections import OrderedDict
from pathlib import Path
from src.layers.affine import Affine
from src.layers.relu import Relu
from src.layers.softmax_with_loss import SoftmaxWithLoss


class TwoLayerNet:
    """2層ニューラルネットワーク"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init_std: float = 0.01,
    ) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # レイヤーの生成
        self.layers: OrderedDict[str, Affine | Relu] = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        y_pred = np.argmax(y, axis=1)
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t

        accuracy = np.sum(y_pred == t_label) / float(x.shape[0])
        return accuracy

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        # forward
        self.loss(x, t)

        # backward
        dout_val = self.last_layer.backward(1.0)

        layers_list = list(self.layers.values())
        layers_list.reverse()
        for layer in layers_list:
            dout_val = layer.backward(dout_val)

        # 勾配を取得
        grads: dict[str, np.ndarray] = {}
        affine1 = self.layers["Affine1"]
        affine2 = self.layers["Affine2"]
        assert isinstance(affine1, Affine) and isinstance(affine2, Affine)
        assert affine1.dW is not None and affine1.db is not None
        assert affine2.dW is not None and affine2.db is not None

        grads["W1"] = affine1.dW
        grads["b1"] = affine1.db
        grads["W2"] = affine2.dW
        grads["b2"] = affine2.db

        return grads

    def save_params(self, file_path: str | Path) -> None:
        """パラメータをファイルに保存"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            file_path,
            W1=self.params["W1"],
            b1=self.params["b1"],
            W2=self.params["W2"],
            b2=self.params["b2"],
        )

    def load_params(self, file_path: str | Path) -> None:
        """パラメータをファイルから読み込み"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {file_path}")

        params = np.load(file_path)
        for key in ("W1", "b1", "W2", "b2"):
            self.params[key] = params[key]

        # レイヤーのパラメータも更新
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
