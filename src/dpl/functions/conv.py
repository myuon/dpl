from dpl import as_variable, Variable, ndarray
import dpl.functions as F


import jax.numpy as jnp
from jax import lax


def conv2d_jax(
    x: jnp.ndarray,  # (N, C, H, W)
    W: jnp.ndarray,  # (OC, C, KH, KW)
    b: jnp.ndarray | None,
    stride=1,
    pad=0,
):
    # stride / pad をタプル化
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(pad, int):
        pad = (pad, pad)

    SH, SW = stride
    PH, PW = pad

    padding = ((PH, PH), (PW, PW))

    dimension_numbers = lax.conv_dimension_numbers(
        lhs_shape=x.shape,
        rhs_shape=W.shape,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )

    y = lax.conv_general_dilated(
        lhs=x,  # (N, C, H, W)
        rhs=W,  # (OC, C, KH, KW)
        window_strides=(SH, SW),
        padding=padding,
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,
        precision=None,
    )  # → (N, OC, OH, OW) = NCHW

    if b is not None:
        y = y + b[None, :, None, None]

    return y


def conv2d(
    x: Variable | ndarray, Q: Variable | ndarray, b, stride=1, pad=0
) -> Variable:
    x, Q = as_variable(x), as_variable(Q)
    Weight = Q

    if (
        isinstance(x.data, jnp.ndarray)
        and isinstance(Weight.data, jnp.ndarray)
        and (b is None or isinstance(b.data, jnp.ndarray))
    ):
        y_data = conv2d_jax(
            x.data,
            Weight.data,
            b.data if b is not None else None,
            stride=stride,
            pad=pad,
        )
        return as_variable(y_data)

    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = stride if isinstance(stride, tuple) else (stride, stride)
    PH, PW = pad if isinstance(pad, tuple) else (pad, pad)
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    col = F.im2col(x, KH, KW, stride, pad)
    Weight = Weight.reshape(OC, -1).transpose()
    t = F.linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)

    return y


def pooling(x: Variable | ndarray, kernel_size: int, stride=1, pad=0) -> Variable:
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = (
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    )
    PH, PW = pad if isinstance(pad, tuple) else (pad, pad)
    SH, SW = stride if isinstance(stride, tuple) else (stride, stride)
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    col = F.im2col(x, KH, KW, stride, pad).reshape(-1, KH * KW)
    y = col.max(axis=1).reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

    return y
