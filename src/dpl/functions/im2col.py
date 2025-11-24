from dpl.core import ndarray, get_array_module, Variable, Function


def _im2col_array(
    x_data: ndarray,
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    pad: int = 0,
) -> ndarray:
    N, C, H, W = x_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    xp = get_array_module(x_data)
    img = xp.pad(x_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def _col2im_array(
    col_data: ndarray,
    input_shape: tuple[int, int, int, int],
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    pad: int = 0,
) -> ndarray:
    """col2imの実際の計算（ndarray版）"""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    xp = get_array_module(col_data)

    # colを元の6次元配列に変形
    # (N*out_h*out_w, C*filter_h*filter_w) -> (N, out_h, out_w, C, filter_h, filter_w)
    col = col_data.reshape(N, out_h, out_w, C, filter_h, filter_w)
    # (N, out_h, out_w, C, filter_h, filter_w) -> (N, C, filter_h, filter_w, out_h, out_w)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    # パディングされた画像サイズ
    img = xp.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    # col2imの逆変換
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # パディングを除去
    if pad > 0:
        return img[:, :, pad : H + pad, pad : W + pad]
    else:
        return img[:, :, :H, :W]


class Im2Col(Function):
    def __init__(self, filter_h: int, filter_w: int, stride: int = 1, pad: int = 0):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.stride = stride
        self.pad = pad

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    def forward(self, *xs: ndarray) -> ndarray:
        (x,) = xs
        self.input_shape = x.shape
        col = _im2col_array(x, self.filter_h, self.filter_w, self.stride, self.pad)
        return col

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        input_shape = tuple(self.input_shape)
        assert len(input_shape) == 4
        gx = col2im(
            gy, input_shape, self.filter_h, self.filter_w, self.stride, self.pad
        )
        return gx


class Col2Im(Function):
    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        filter_h: int,
        filter_w: int,
        stride: int = 1,
        pad: int = 0,
    ):
        self.input_shape = input_shape
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.stride = stride
        self.pad = pad

    def apply(self, col: Variable) -> Variable:
        result = super().__call__(col)
        assert isinstance(result, Variable)
        return result

    def forward(self, *cols: ndarray) -> ndarray:
        (col,) = cols
        self.col_shape = col.shape
        img = _col2im_array(
            col, self.input_shape, self.filter_h, self.filter_w, self.stride, self.pad
        )
        return img

    def backward(self, *gys: Variable) -> Variable:
        (gy,) = gys
        gcol = im2col(gy, self.filter_h, self.filter_w, self.stride, self.pad)
        return gcol


def im2col(
    x: Variable,
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    pad: int = 0,
) -> Variable:
    return Im2Col(filter_h, filter_w, stride, pad).apply(x)


def col2im(
    col: Variable,
    input_shape: tuple[int, int, int, int],
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    pad: int = 0,
) -> Variable:
    return Col2Im(input_shape, filter_h, filter_w, stride, pad).apply(col)
