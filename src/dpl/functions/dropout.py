from dpl import (
    Variable,
    ndarray,
    Config,
    as_variable,
    get_random_module,
    get_array_module,
)


def dropout(x: Variable | ndarray, dropout_ratio=0.5):
    x = as_variable(x)

    if Config.train:
        xr = get_random_module(x)
        xp = get_array_module(x.data)
        mask = xr.randn(*x.shape) > dropout_ratio
        scale = xp.asarray(1.0 - dropout_ratio, dtype=x.dtype)
        y = x * mask / scale
        return y
    else:
        return x
