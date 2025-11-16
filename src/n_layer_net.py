import numpy as np
from collections import OrderedDict
from typing import Callable, Optional
from layers.affine import Affine
from layers.relu import Relu
from layers.batch_normalization import BatchNormalization
from layers.softmax_with_loss import SoftmaxWithLoss
from computational_graph import get_global_recorder


class NLayerNet:
    """N層ニューラルネットワーク

    任意の層数のニューラルネットワークを構築できる。
    各層の間にReLU活性化関数を挿入する。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        hidden_layer_num: int = 1,
        weight_init_std: float = 0.01,
        weight_initializer: Optional[Callable[[int, int], np.ndarray]] = None,
        use_batchnorm: bool = False,
    ) -> None:
        """
        Args:
            input_size: 入力層のサイズ
            hidden_size: 隠れ層のサイズ
            output_size: 出力層のサイズ
            hidden_layer_num: 隠れ層の数
            weight_init_std: 重みの初期化時の標準偏差（weight_initializerがNoneの場合に使用）
            weight_initializer: 重み初期化関数。(n_in, n_out) -> np.ndarray の形式
            use_batchnorm: Batch Normalizationを使用するかどうか
        """
        self.hidden_layer_num = hidden_layer_num
        self.use_batchnorm = use_batchnorm
        self.params: dict[str, np.ndarray] = {}

        # 重み初期化関数の設定
        if weight_initializer is None:
            # デフォルトの初期化関数
            def default_init(n_in: int, n_out: int) -> np.ndarray:
                return weight_init_std * np.random.randn(n_in, n_out)

            weight_init_fn: Callable[[int, int], np.ndarray] = default_init
        else:
            weight_init_fn = weight_initializer

        # パラメータの初期化
        # 第1層（入力層 -> 最初の隠れ層）
        self.params["W1"] = weight_init_fn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)

        # 中間の隠れ層
        for i in range(2, hidden_layer_num + 1):
            self.params[f"W{i}"] = weight_init_fn(hidden_size, hidden_size)
            self.params[f"b{i}"] = np.zeros(hidden_size)

        # 最終層（最後の隠れ層 -> 出力層）
        final_layer_idx = hidden_layer_num + 1
        self.params[f"W{final_layer_idx}"] = weight_init_fn(hidden_size, output_size)
        self.params[f"b{final_layer_idx}"] = np.zeros(output_size)

        # Batch Normalizationのパラメータ初期化
        if use_batchnorm:
            for i in range(1, hidden_layer_num + 1):
                self.params[f"gamma{i}"] = np.ones(hidden_size)
                self.params[f"beta{i}"] = np.zeros(hidden_size)

        # レイヤーの生成
        self.layers: OrderedDict[str, Affine | Relu | BatchNormalization] = (
            OrderedDict()
        )

        # 隠れ層の構築（Affine + [BatchNorm] + ReLU）
        for i in range(1, hidden_layer_num + 1):
            self.layers[f"Affine{i}"] = Affine(
                self.params[f"W{i}"], self.params[f"b{i}"]
            )

            if use_batchnorm:
                self.layers[f"BatchNorm{i}"] = BatchNormalization(
                    self.params[f"gamma{i}"], self.params[f"beta{i}"]
                )

            self.layers[f"Relu{i}"] = Relu()

        # 出力層（Affineのみ）
        self.layers[f"Affine{final_layer_idx}"] = Affine(
            self.params[f"W{final_layer_idx}"], self.params[f"b{final_layer_idx}"]
        )

        self.last_layer = SoftmaxWithLoss()

    def predict(
        self, x: np.ndarray, record_activations: bool = False, train_flg: bool = False
    ) -> np.ndarray:
        """予測を実行

        Args:
            x: 入力データ
            record_activations: Trueの場合、各層の活性化値を記録する
            train_flg: 学習時はTrue、推論時はFalse（Batch Normalization用）

        Returns:
            予測結果
        """
        if record_activations:
            self.activations: dict[str, np.ndarray] = {}

        recorder = get_global_recorder()

        # 入力ノードを記録
        prev_node = -1
        if recorder.enabled:
            prev_node = recorder.add_node("Input", shape=x.shape, node_type="data")

        for layer_name, layer in self.layers.items():
            # レイヤーのforward処理
            if isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

            # 計算グラフに記録
            if recorder.enabled:
                current_node = recorder.add_node(
                    layer_name, shape=x.shape, node_type="operation"
                )
                recorder.add_edge(prev_node, current_node, edge_type="forward")
                prev_node = current_node

            if record_activations:
                self.activations[layer_name] = x.copy()

        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x, train_flg=True)
        return float(self.last_layer.forward(y, t))

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x, train_flg=False)
        y_pred = np.argmax(y, axis=1)
        t_label = np.argmax(t, axis=1) if t.ndim != 1 else t

        accuracy = np.sum(y_pred == t_label) / float(x.shape[0])
        return float(accuracy)

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        # forward
        self.loss(x, t)

        recorder = get_global_recorder()

        # backward
        dout_val = self.last_layer.backward(1.0)

        # 計算グラフに損失関数ノードを追加
        loss_node = -1
        if recorder.enabled:
            loss_node = recorder.add_node(
                "SoftmaxWithLoss", shape=dout_val.shape, node_type="operation"
            )

        layers_list = list(self.layers.values())
        layer_names = list(self.layers.keys())
        layers_list.reverse()
        layer_names.reverse()

        # backwardの記録用
        prev_node = loss_node

        for layer_name, layer in zip(layer_names, layers_list):
            dout_val = layer.backward(dout_val)

            # 計算グラフに記録（backward）
            if recorder.enabled:
                current_node = recorder.add_node(
                    f"{layer_name}_backward",
                    shape=dout_val.shape,
                    node_type="operation",
                )
                recorder.add_edge(prev_node, current_node, edge_type="backward")
                prev_node = current_node

        # 勾配を取得
        grads: dict[str, np.ndarray] = {}

        # 全ての層の勾配を取得
        for i in range(1, self.hidden_layer_num + 2):
            affine_layer = self.layers[f"Affine{i}"]
            assert isinstance(affine_layer, Affine)
            assert affine_layer.dW is not None and affine_layer.db is not None

            grads[f"W{i}"] = affine_layer.dW
            grads[f"b{i}"] = affine_layer.db

        # Batch Normalizationの勾配を取得
        if self.use_batchnorm:
            for i in range(1, self.hidden_layer_num + 1):
                bn_layer = self.layers[f"BatchNorm{i}"]
                assert isinstance(bn_layer, BatchNormalization)
                assert bn_layer.dgamma is not None and bn_layer.dbeta is not None

                grads[f"gamma{i}"] = bn_layer.dgamma
                grads[f"beta{i}"] = bn_layer.dbeta

        return grads
