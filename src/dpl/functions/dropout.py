from dpl import Variable, ndarray, Config, as_variable, get_array_module


def dropout(x: Variable | ndarray, dropout_ratio=0.5):
    x = as_variable(x)

    if Config.train:
        xp = get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio, dtype=x.dtype)
        y = x * mask / scale
        return y
    else:
        return x
