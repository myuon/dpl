from dpl import as_variable, Variable, ndarray
from dpl.core import Function
import dpl.functions as F


import jax.numpy as jnp
from jax import lax


def conv2d_jax_forward(
    x: jnp.ndarray,  # (N, C, H, W)
    W: jnp.ndarray,  # (OC, C, KH, KW)
    b: jnp.ndarray | None,
    stride: tuple[int, int],
    pad: tuple[int, int],
):
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


class Conv2dJax(Function):
    """JAX-based Conv2d with proper backward support."""

    def __init__(self, stride: tuple[int, int], pad: tuple[int, int], has_bias: bool):
        self.stride = stride
        self.pad = pad
        self.has_bias = has_bias

    def forward(self, *xs: ndarray) -> ndarray:
        (x, W) = xs[0], xs[1]
        b = xs[2] if self.has_bias else None
        assert isinstance(x, jnp.ndarray)
        assert isinstance(W, jnp.ndarray)
        if b is not None:
            assert isinstance(b, jnp.ndarray)

        y = conv2d_jax_forward(
            x,
            W,
            b if b is not None else None,
            self.stride,
            self.pad,
        )
        return y

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        (gy,) = gys
        x, W = self.inputs[0], self.inputs[1]
        b = self.inputs[2] if self.has_bias else None

        SH, SW = self.stride
        PH, PW = self.pad

        # Get shapes
        N, C, H, W_dim = x.shape
        OC, _, KH, KW = W.shape
        _, _, OH, OW = gy.shape

        # Gradient for x: transposed convolution (deconvolution)
        # Use lax.conv_transpose_general_dilated
        gy_data = gy.data_required
        W_data = W.data_required

        padding_for_transpose = ((KH - PH - 1, KH - PH - 1), (KW - PW - 1, KW - PW - 1))

        dimension_numbers = lax.conv_dimension_numbers(
            lhs_shape=gy_data.shape,
            rhs_shape=W_data.shape,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        # For dx: convolve gy with W (transposed)
        # W shape: (OC, C, KH, KW) -> need to transpose to (C, OC, KH, KW) and flip
        W_transposed = jnp.transpose(W_data, (1, 0, 2, 3))  # (C, OC, KH, KW)
        W_flipped = jnp.flip(W_transposed, axis=(2, 3))  # flip kernel

        dimension_numbers_dx = lax.conv_dimension_numbers(
            lhs_shape=gy_data.shape,
            rhs_shape=W_flipped.shape,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        gx_data = lax.conv_general_dilated(
            lhs=gy_data,  # (N, OC, OH, OW)
            rhs=W_flipped,  # (C, OC, KH, KW)
            window_strides=(1, 1),
            padding=padding_for_transpose,
            lhs_dilation=(SH, SW),
            rhs_dilation=None,
            dimension_numbers=dimension_numbers_dx,
            feature_group_count=1,
            precision=None,
        )

        # Crop or pad gx to match input size
        _, _, gx_H, gx_W = gx_data.shape
        if gx_H > H or gx_W > W_dim:
            # Crop
            start_h = (gx_H - H) // 2
            start_w = (gx_W - W_dim) // 2
            gx_data = gx_data[:, :, start_h : start_h + H, start_w : start_w + W_dim]
        elif gx_H < H or gx_W < W_dim:
            # Pad
            pad_h = (H - gx_H) // 2
            pad_w = (W_dim - gx_W) // 2
            gx_data = jnp.pad(
                gx_data,
                (
                    (0, 0),
                    (0, 0),
                    (pad_h, H - gx_H - pad_h),
                    (pad_w, W_dim - gx_W - pad_w),
                ),
            )

        gx = as_variable(gx_data)

        # Gradient for W: correlation of x and gy
        x_data = x.data_required

        # Pad input
        if PH > 0 or PW > 0:
            x_padded = jnp.pad(x_data, ((0, 0), (0, 0), (PH, PH), (PW, PW)))
        else:
            x_padded = x_data

        # x_padded: (N, C, H+2*PH, W+2*PW)
        # gy: (N, OC, OH, OW)
        # gW should be: (OC, C, KH, KW)

        # Reshape for batch correlation
        # x_padded: (N, C, H', W') -> transpose to (C, N, H', W')
        x_t = jnp.transpose(x_padded, (1, 0, 2, 3))  # (C, N, H', W')

        # gy: (N, OC, OH, OW) -> transpose to (OC, N, OH, OW)
        gy_t = jnp.transpose(gy_data, (1, 0, 2, 3))  # (OC, N, OH, OW)

        dimension_numbers_gW = lax.conv_dimension_numbers(
            lhs_shape=x_t.shape,
            rhs_shape=gy_t.shape,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        gW_data = lax.conv_general_dilated(
            lhs=x_t,  # (C, N, H', W')
            rhs=gy_t,  # (OC, N, OH, OW)
            window_strides=(1, 1),
            padding="VALID",
            lhs_dilation=(1, 1),
            rhs_dilation=(SH, SW),
            dimension_numbers=dimension_numbers_gW,
            feature_group_count=1,
            precision=None,
        )  # -> (C, OC, KH, KW)

        # Transpose back to (OC, C, KH, KW)
        gW_data = jnp.transpose(gW_data, (1, 0, 2, 3))

        # Ensure gW has correct shape
        if gW_data.shape != W.shape:
            # Crop if needed
            gW_data = gW_data[:, :, :KH, :KW]

        gW = as_variable(gW_data)

        if self.has_bias:
            # Gradient for b: sum over N, H, W
            gb_data = jnp.sum(gy_data, axis=(0, 2, 3))
            gb = as_variable(gb_data)
            return gx, gW, gb
        else:
            return gx, gW


def conv2d(
    x: Variable | ndarray, Q: Variable | ndarray, b, stride=1, pad=0
) -> Variable:
    x, Q = as_variable(x), as_variable(Q)
    Weight = Q

    # Normalize stride and pad to tuples
    stride_tuple = stride if isinstance(stride, tuple) else (stride, stride)
    pad_tuple = pad if isinstance(pad, tuple) else (pad, pad)

    if (
        isinstance(x.data, jnp.ndarray)
        and isinstance(Weight.data, jnp.ndarray)
        and (b is None or isinstance(b.data, jnp.ndarray))
    ):
        has_bias = b is not None
        func = Conv2dJax(stride_tuple, pad_tuple, has_bias)
        if has_bias:
            b = as_variable(b)
            result = func(x, Weight, b)
        else:
            result = func(x, Weight)
        assert isinstance(result, Variable)
        return result

    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = stride_tuple
    PH, PW = pad_tuple
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
