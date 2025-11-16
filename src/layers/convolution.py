import numpy as np


def im2col(input_data: np.ndarray, filter_h: int, filter_w: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """画像を行列に変換する（畳み込みの高速化用）

    Args:
        input_data: 入力データ (N, C, H, W)
        filter_h: フィルタの高さ
        filter_w: フィルタの幅
        stride: ストライド
        pad: パディング

    Returns:
        2次元配列に変換した画像データ
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # パディング
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # 出力用の配列を準備
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # (N, C, filter_h, filter_w, out_h, out_w) -> (N*out_h*out_w, C*filter_h*filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col: np.ndarray, input_shape: tuple, filter_h: int, filter_w: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """行列を画像に戻す（im2colの逆変換）

    Args:
        col: 2次元配列
        input_shape: 入力データの形状 (N, C, H, W)
        filter_h: フィルタの高さ
        filter_w: フィルタの幅
        stride: ストライド
        pad: パディング

    Returns:
        画像データ
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # (N*out_h*out_w, C*filter_h*filter_w) -> (N, out_h, out_w, C, filter_h, filter_w)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # パディングを含む画像サイズ
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # パディング部分を除去
    return img[:, :, pad:H + pad, pad:W + pad]


class Convolution:
    """畳み込み層

    4次元配列のデータを処理する：
    - 入力: (N, C, H, W) - N: バッチサイズ, C: チャンネル数, H: 高さ, W: 幅
    - 出力: (N, FN, OH, OW) - FN: フィルタ数, OH: 出力高さ, OW: 出力幅
    """

    def __init__(
        self,
        W: np.ndarray,
        b: np.ndarray,
        stride: int = 1,
        pad: int = 0
    ) -> None:
        """
        Args:
            W: フィルタの重み (FN, C, FH, FW)
                FN: フィルタ数, C: チャンネル数, FH: フィルタ高さ, FW: フィルタ幅
            b: バイアス (FN,)
            stride: ストライド
            pad: パディング
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # backward用
        self.x: np.ndarray | None = None
        self.col: np.ndarray | None = None
        self.col_W: np.ndarray | None = None

        # 勾配
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力 (N, C, H, W)

        Returns:
            出力 (N, FN, OH, OW)
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        # 出力サイズの計算
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        # im2colで画像を行列に変換
        col = im2col(x, FH, FW, self.stride, self.pad)

        # フィルタを2次元配列に変換 (FN, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T  # (C*FH*FW, FN)

        # 行列の積を計算
        out = np.dot(col, col_W) + self.b  # (N*out_h*out_w, FN)

        # 形状を整形 (N, out_h, out_w, FN) -> (N, FN, out_h, out_w)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # backward用に保存
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """逆伝播

        Args:
            dout: 出力側の勾配 (N, FN, OH, OW)

        Returns:
            入力側の勾配 (N, C, H, W)
        """
        assert self.x is not None
        assert self.col is not None
        assert self.col_W is not None

        FN, C, FH, FW = self.W.shape

        # (N, FN, OH, OW) -> (N, OH, OW, FN) -> (N*OH*OW, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # バイアスの勾配
        self.db = np.sum(dout, axis=0)

        # 重みの勾配
        self.dW = np.dot(self.col.T, dout)  # (C*FH*FW, FN)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 入力の勾配
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
