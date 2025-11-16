import numpy as np
from collections import OrderedDict
from typing import Optional
from layers.convolution import Convolution
from layers.relu import Relu
from layers.pooling import Pooling
from layers.affine import Affine
from layers.dropout import Dropout
from layers.softmax_with_loss import SoftmaxWithLoss
from computational_graph import get_global_recorder


class SimpleCNN:
    """シンプルなCNNモデル

    アーキテクチャ:
    Input -> (Conv -> ReLU -> Conv -> ReLU -> Pooling) * 3
          -> Affine -> ReLU -> Dropout -> Affine -> Dropout -> Softmax
    """

    def __init__(
        self,
        input_dim: tuple[int, int, int] = (1, 28, 28),
        conv_param: Optional[dict] = None,
        hidden_size: int = 100,
        output_size: int = 10,
        weight_init_std: float = 0.01,
        dropout_ratio: float = 0.5,
    ) -> None:
        # backward用に保存
        self.pool_output_shape: tuple[int, int, int, int] | None = None
        """
        Args:
            input_dim: 入力データの形状 (C, H, W)
            conv_param: 畳み込み層のパラメータ
                {
                    'filter_num': フィルタ数のリスト (3要素),
                    'filter_size': フィルタサイズ,
                    'pad': パディング,
                    'stride': ストライド
                }
            hidden_size: 全結合層の隠れ層サイズ
            output_size: 出力サイズ（クラス数）
            weight_init_std: 重みの初期化時の標準偏差
            dropout_ratio: ドロップアウト率
        """
        if conv_param is None:
            conv_param = {
                'filter_num': [16, 32, 64],
                'filter_size': 3,
                'pad': 1,
                'stride': 1
            }

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_channel, input_h, input_w = input_dim

        # 各畳み込み層のチャンネル数
        # 各ブロックで2つのConv層があるので、チャンネル数は
        # [input_channel, filter_num[0], filter_num[0], filter_num[1], filter_num[1], filter_num[2], filter_num[2]]
        channels = [input_channel]
        for fn in filter_num:
            channels.extend([fn, fn])

        self.params: dict[str, np.ndarray] = {}

        # 畳み込み層のパラメータ初期化
        # (Conv -> ReLU -> Conv -> ReLU -> Pooling) * 3
        conv_output_h = input_h
        conv_output_w = input_w

        for block_idx in range(3):
            for conv_idx in range(2):
                layer_idx = block_idx * 2 + conv_idx + 1
                in_channels = channels[layer_idx - 1]
                out_channels = channels[layer_idx]

                # Heの初期化（ReLU用）
                self.params[f'W_conv{layer_idx}'] = weight_init_std * np.random.randn(
                    out_channels, in_channels, filter_size, filter_size
                ) * np.sqrt(2.0 / (in_channels * filter_size * filter_size))
                self.params[f'b_conv{layer_idx}'] = np.zeros(out_channels)

            # プーリング層の後のサイズを計算
            conv_output_h = conv_output_h // 2
            conv_output_w = conv_output_w // 2

        # 全結合層のパラメータ初期化
        fc_input_size = filter_num[-1] * conv_output_h * conv_output_w

        self.params['W_fc1'] = weight_init_std * np.random.randn(
            fc_input_size, hidden_size
        ) * np.sqrt(2.0 / fc_input_size)
        self.params['b_fc1'] = np.zeros(hidden_size)

        self.params['W_fc2'] = weight_init_std * np.random.randn(
            hidden_size, output_size
        ) * np.sqrt(2.0 / hidden_size)
        self.params['b_fc2'] = np.zeros(output_size)

        # レイヤーの生成
        self.layers: OrderedDict[str, Convolution | Relu | Pooling | Affine | Dropout] = OrderedDict()

        # (Conv -> ReLU -> Conv -> ReLU -> Pooling) * 3
        for block_idx in range(3):
            # Conv -> ReLU (1st conv in block)
            layer_idx = block_idx * 2 + 1
            self.layers[f'Conv{layer_idx}'] = Convolution(
                self.params[f'W_conv{layer_idx}'],
                self.params[f'b_conv{layer_idx}'],
                stride=filter_stride,
                pad=filter_pad
            )
            self.layers[f'Relu{layer_idx}'] = Relu()

            # Conv -> ReLU (2nd conv in block)
            layer_idx = block_idx * 2 + 2
            self.layers[f'Conv{layer_idx}'] = Convolution(
                self.params[f'W_conv{layer_idx}'],
                self.params[f'b_conv{layer_idx}'],
                stride=filter_stride,
                pad=filter_pad
            )
            self.layers[f'Relu{layer_idx}'] = Relu()

            # Pooling
            self.layers[f'Pool{block_idx + 1}'] = Pooling(
                pool_h=2, pool_w=2, stride=2, pad=0
            )

        # Affine -> ReLU -> Dropout
        self.layers['Affine1'] = Affine(self.params['W_fc1'], self.params['b_fc1'])
        self.layers['Relu_fc1'] = Relu()
        self.layers['Dropout1'] = Dropout(dropout_ratio=dropout_ratio)

        # Affine -> Dropout
        self.layers['Affine2'] = Affine(self.params['W_fc2'], self.params['b_fc2'])
        self.layers['Dropout2'] = Dropout(dropout_ratio=dropout_ratio)

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray, train_flg: bool = False) -> np.ndarray:
        """予測を実行

        Args:
            x: 入力データ (N, C, H, W)
            train_flg: 学習時はTrue、推論時はFalse

        Returns:
            予測結果 (N, output_size)
        """
        recorder = get_global_recorder()

        # 入力ノードを記録
        prev_node = -1
        if recorder.enabled:
            prev_node = recorder.add_node("Input", shape=x.shape, node_type="data")

        for layer_name, layer in self.layers.items():
            # Affine層の前にflatten
            if isinstance(layer, Affine) and x.ndim == 4:
                self.pool_output_shape = x.shape  # backward用に保存
                x = x.reshape(x.shape[0], -1)

            # Dropout層は train_flg を渡す
            if isinstance(layer, Dropout):
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

        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        """損失関数の値を計算

        Args:
            x: 入力データ
            t: 教師ラベル

        Returns:
            損失
        """
        y = self.predict(x, train_flg=True)
        return float(self.last_layer.forward(y, t))

    def accuracy(self, x: np.ndarray, t: np.ndarray, batch_size: int = 100) -> float:
        """精度を計算

        Args:
            x: 入力データ
            t: 教師ラベル
            batch_size: バッチサイズ（メモリ節約のため）

        Returns:
            精度
        """
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:i + batch_size]
            t_batch = t[i:i + batch_size]
            y = self.predict(x_batch, train_flg=False)
            y_pred = np.argmax(y, axis=1)
            acc += np.sum(y_pred == t_batch)

        return float(acc / x.shape[0])

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        """勾配を計算

        Args:
            x: 入力データ
            t: 教師ラベル

        Returns:
            各パラメータの勾配
        """
        # forward
        self.loss(x, t)

        recorder = get_global_recorder()

        # backward
        dout = self.last_layer.backward(1.0)

        # 計算グラフに損失関数ノードを追加
        loss_node = -1
        if recorder.enabled:
            loss_node = recorder.add_node(
                "SoftmaxWithLoss", shape=dout.shape, node_type="operation"
            )

        layers_list = list(self.layers.values())
        layer_names = list(self.layers.keys())
        layers_list.reverse()
        layer_names.reverse()

        # backwardの記録用
        prev_node = loss_node

        for layer_name, layer in zip(layer_names, layers_list):
            dout = layer.backward(dout)

            # Affine1層の後にreshape（4次元に戻す）
            if layer_name == "Affine1" and self.pool_output_shape is not None:
                dout = dout.reshape(self.pool_output_shape)

            # 計算グラフに記録（backward）
            if recorder.enabled:
                current_node = recorder.add_node(
                    f"{layer_name}_backward",
                    shape=dout.shape,
                    node_type="operation",
                )
                recorder.add_edge(prev_node, current_node, edge_type="backward")
                prev_node = current_node

        # 勾配を取得
        grads: dict[str, np.ndarray] = {}

        # 畳み込み層の勾配
        for i in range(1, 7):  # Conv1〜Conv6
            conv_layer = self.layers[f'Conv{i}']
            assert isinstance(conv_layer, Convolution)
            assert conv_layer.dW is not None and conv_layer.db is not None
            grads[f'W_conv{i}'] = conv_layer.dW
            grads[f'b_conv{i}'] = conv_layer.db

        # 全結合層の勾配
        for i in range(1, 3):  # Affine1, Affine2
            affine_layer = self.layers[f'Affine{i}']
            assert isinstance(affine_layer, Affine)
            assert affine_layer.dW is not None and affine_layer.db is not None
            grads[f'W_fc{i}'] = affine_layer.dW
            grads[f'b_fc{i}'] = affine_layer.db

        return grads
