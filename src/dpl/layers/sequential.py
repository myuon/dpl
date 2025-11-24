from dpl.core import Variable
from dpl.layers.layer import Layer
from typing import Callable


class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers: list[Layer] = []

        for i, layer in enumerate(layers):
            # Register layers as attributes so they're tracked by Layer's __setattr__
            if isinstance(layer, Layer):
                setattr(self, f"l{i}", layer)
                self.layers.append(getattr(self, f"l{i}"))
            else:
                # Allow callables (like lambda x: F.relu(x))
                self.layers.append(layer)

    def forward(self, *inputs: Variable) -> Variable:
        result = inputs
        for layer in self.layers:
            result = layer(*result)

        return result if not isinstance(result, tuple) else result[0]

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]
