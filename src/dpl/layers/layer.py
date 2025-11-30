import weakref
from dpl.core import Variable


class Parameter(Variable):
    pass


class Layer:
    def __init__(self):
        self._params: set[str] = set()

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: Variable) -> Variable:
        output = self.forward(*inputs)

        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = (
            [weakref.ref(wr) for wr in output]
            if isinstance(output, tuple)
            else [weakref.ref(output)]
        )
        return output

    def forward(self, *inputs: Variable) -> Variable:
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            value = self.__dict__[name]

            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Layer):
                yield from value.params()

    def _flatten_params(self, params_dict, parent_key=[]):
        for name in self._params:
            value = self.__dict__[name]
            key = parent_key.copy()
            key.append(name)

            if isinstance(value, Layer):
                value._flatten_params(params_dict, key)
            else:
                params_dict["/".join(key)] = value

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def save_weights(self, path: str) -> None:
        import os
        import numpy as np

        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {k: v.data for k, v in params_dict.items()}

        try:
            np.savez_compressed(path, **array_dict)
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path: str) -> None:
        import numpy as np

        loaded = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)

        for k, param in params_dict.items():
            if k in loaded:
                param.data = loaded[k]
            else:
                raise KeyError(f"Key '{k}' not found in the loaded weights.")


class StatefulLayer(Layer):
    """Base class for layers that maintain internal state."""

    def reset_state(self):
        """Reset layer state. Override in subclasses."""
        pass


class UnaryLayer(Layer):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result


class BinaryLayer(Layer):
    def apply(self, x0: Variable, x1: Variable) -> Variable:
        result = super().__call__(x0, x1)
        assert isinstance(result, Variable)
        return result
