from dpl.core import ndarray, get_array_module, Variable, UnaryFunction
from jax import jit
import jax.numpy as jnp
from functools import partial


def _im2col_array_cpu(
    img: ndarray,
    N: int,
    C: int,
    filter_h: int,
    filter_w: int,
    out_h: int,
    out_w: int,
    stride: int,
    xp,
) -> ndarray:
    """CPU (NumPy) version of im2col using in-place assignment"""
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


@partial(jit, static_argnames=("filter_h", "filter_w", "stride", "pad"))
def _im2col_array_jax(
    img,
    filter_h: int,
    filter_w: int,
    stride: int,
    pad: int,
):
    N, C, H, W = img.shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # ここで out_h, out_w は Python int なので OK
    oh = jnp.arange(out_h)[:, None, None, None]  # (out_h, 1,      1,      1)
    ow = jnp.arange(out_w)[None, :, None, None]  # (1,      out_w, 1,      1)
    kh = jnp.arange(filter_h)[None, None, :, None]
    kw = jnp.arange(filter_w)[None, None, None, :]

    iy = oh * stride + kh
    ix = ow * stride + kw

    col = img[:, :, iy, ix]  # (N, C, out_h, out_w, filter_h, filter_w)
    col = col.transpose(0, 2, 3, 1, 4, 5)  # (N, out_h, out_w, C, filter_h, filter_w)
    col = col.reshape(N * out_h * out_w, -1)  # (N*out_h*out_w, C*filter_h*filter_w)
    return col


def _im2col_array(
    x_data: ndarray,
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    pad: int = 0,
) -> ndarray:
    import jax.numpy as jnp

    N, C, H, W = x_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    xp = get_array_module(x_data)
    img = xp.pad(x_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    if isinstance(x_data, jnp.ndarray):
        # imgは既にパディング済みなのでpad=0を渡す
        return _im2col_array_jax(img, filter_h, filter_w, stride, 0)
    else:
        return _im2col_array_cpu(
            img, N, C, filter_h, filter_w, out_h, out_w, stride, xp
        )


def _col2im_array_cpu(
    col: ndarray,
    N: int,
    C: int,
    H: int,
    W: int,
    filter_h: int,
    filter_w: int,
    out_h: int,
    out_w: int,
    stride: int,
    pad: int,
    xp,
) -> ndarray:
    """CPU (NumPy) version of col2im using in-place addition"""
    img = xp.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

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


@partial(
    jit, static_argnames=("N", "C", "H", "W", "filter_h", "filter_w", "stride", "pad")
)
def _col2im_array_jax(
    col: ndarray,
    N: int,
    C: int,
    H: int,
    W: int,
    filter_h: int,
    filter_w: int,
    stride: int,
    pad: int,
):
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    H_padded = H + 2 * pad + stride - 1
    W_padded = W + 2 * pad + stride - 1

    img = jnp.zeros((N, C, H_padded, W_padded), dtype=col.dtype)

    y = jnp.arange(filter_h)[:, None, None, None]
    x = jnp.arange(filter_w)[None, :, None, None]
    oh = jnp.arange(out_h)[None, None, :, None]
    ow = jnp.arange(out_w)[None, None, None, :]

    iy = y + stride * oh
    ix = x + stride * ow

    img = img.at[:, :, iy, ix].add(col)

    if pad > 0:
        return img[:, :, pad : H + pad, pad : W + pad]
    else:
        return img[:, :, :H, :W]


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

    import jax.numpy as jnp

    if isinstance(col_data, jnp.ndarray):
        return _col2im_array_jax(col, N, C, H, W, filter_h, filter_w, stride, pad)
    else:
        return _col2im_array_cpu(
            col, N, C, H, W, filter_h, filter_w, out_h, out_w, stride, pad, xp
        )


class Im2Col(UnaryFunction):
    def __init__(self, filter_h: int, filter_w: int, stride: int = 1, pad: int = 0):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.stride = stride
        self.pad = pad

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


class Col2Im(UnaryFunction):
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
