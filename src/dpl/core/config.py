import contextlib


@contextlib.contextmanager
def use_config(name: str, value: object):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


class Config:
    enable_backprop = True
    train = True


def no_grad():
    return use_config("enable_backprop", False)


def test_mode():
    return use_config("train", False)
