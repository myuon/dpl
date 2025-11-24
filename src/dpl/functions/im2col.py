from dpl import ndarray, get_array_module, Variable, as_variable


def im2col(
    input_data: ndarray | Variable,
    filter_h: int,
    filter_w: int,
    stride: int = 1,
    pad: int = 0,
) -> ndarray:
    x = as_variable(input_data)

    N, C, H, W = x.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # パディング
    xp = get_array_module(x.data)
    img = xp.pad(x.data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    # 出力用の配列を準備
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # (N, C, filter_h, filter_w, out_h, out_w) -> (N*out_h*out_w, C*filter_h*filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col
