from dpl.core import Variable
from dpl.layers.layer import Layer, StatefulLayer
from typing import Callable


class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers: list[Layer | Callable] = []

        for i, layer in enumerate(layers):
            if isinstance(layer, Layer):
                setattr(self, f"l{i}", layer)
                self.layers.append(getattr(self, f"l{i}"))
            else:
                self.layers.append(layer)

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def apply(self, *xs: Variable) -> Variable:
        result = super().__call__(*xs)
        assert isinstance(result, Variable)
        return result

    def reset_state(self):
        """Reset state for all stateful layers."""
        for layer in self.layers:
            if isinstance(layer, StatefulLayer):
                layer.reset_state()

    def __getitem__(self, name: str) -> Layer:
        return getattr(self, name)
