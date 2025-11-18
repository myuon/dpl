import numpy as np

from src.layers.convolution import im2col


class Pooling:
    """プーリング層（Max Pooling）

    4次元配列のデータを処理する：
    - 入力: (N, C, H, W) - N: バッチサイズ, C: チャンネル数, H: 高さ, W: 幅
    - 出力: (N, C, OH, OW) - OH: 出力高さ, OW: 出力幅

    チャンネル数は変わらず、空間方向のサイズだけが縮小される。
    """

    def __init__(self, pool_h: int, pool_w: int, stride: int = 1, pad: int = 0) -> None:
        """
        Args:
            pool_h: プーリング領域の高さ
            pool_w: プーリング領域の幅
            stride: ストライド
            pad: パディング
        """
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # backward用
        self.x: np.ndarray | None = None
        self.arg_max: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力 (N, C, H, W)

        Returns:
            出力 (N, C, OH, OW)
        """
        N, C, H, W = x.shape

        # 出力サイズの計算
        out_h = (H + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.pad - self.pool_w) // self.stride + 1

        # im2colとreshapeを一度に実行
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad).reshape(
            -1, self.pool_h * self.pool_w
        )

        # argmaxとmaxを統合: argmaxの結果を使ってmaxの値を取得
        arg_max = np.argmax(col, axis=1)
        out = col[np.arange(col.shape[0]), arg_max]

        # 形状を整形 (N*out_h*out_w*C,) -> (N, out_h, out_w, C) -> (N, C, out_h, out_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # backward用に保存
        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """逆伝播

        Args:
            dout: 出力側の勾配 (N, C, OH, OW)

        Returns:
            入力側の勾配 (N, C, H, W)
        """
        assert self.x is not None
        assert self.arg_max is not None

        N, C, H, W = self.x.shape
        out_h, out_w = dout.shape[2], dout.shape[3]

        # (N, C, out_h, out_w) -> (N, out_h, out_w, C) -> (N*out_h*out_w*C,)
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1)

        # 勾配の分配用の配列 (N*out_h*out_w*C, pool_h*pool_w)
        dmax = np.zeros((dout_flat.size, self.pool_h * self.pool_w))

        # 最大値だった位置にだけ勾配を流す
        dmax[np.arange(self.arg_max.size), self.arg_max] = dout_flat

        # (N*out_h*out_w*C, pool_h*pool_w) -> (N, out_h, out_w, C, pool_h, pool_w)
        dmax = dmax.reshape(N, out_h, out_w, C, self.pool_h, self.pool_w)

        # col2im的な逆変換
        # パディングを含む入力サイズ
        dx = np.zeros(
            (
                N,
                C,
                H + 2 * self.pad + self.stride - 1,
                W + 2 * self.pad + self.stride - 1,
            )
        )

        for y in range(self.pool_h):
            y_max = y + self.stride * out_h
            for x in range(self.pool_w):
                x_max = x + self.stride * out_w
                # dmax: (N, out_h, out_w, C, pool_h, pool_w)
                # dx: (N, C, H, W) なので、transposeして次元を合わせる
                dx[:, :, y : y_max : self.stride, x : x_max : self.stride] += dmax[
                    :, :, :, :, y, x
                ].transpose(0, 3, 1, 2)

        # パディング部分を除去
        if self.pad > 0:
            dx = dx[:, :, self.pad : H + self.pad, self.pad : W + self.pad]
        else:
            dx = dx[:, :, :H, :W]

        return dx
