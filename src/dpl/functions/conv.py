from dpl import as_variable, Variable, ndarray
import dpl.functions as F


def conv2d(
    x: Variable | ndarray, Q: Variable | ndarray, b, stride=1, pad=0
) -> Variable:
    x, Q = as_variable(x), as_variable(Q)

    Weight = Q
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
