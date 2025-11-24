from dpl.core import Variable
import dpl.functions as F
import dpl.layers as L
from dpl.models.model import Model


class MLP(Model):
    def __init__(self, out_sizes: list[int], activation=F.sigmoid) -> None:
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(out_sizes):
            layer = L.Linear(out_size)
            setattr(self, f"l{i+1}", layer)
            self.layers.append(layer)

    def apply(self, x: Variable) -> Variable:
        out = super().__call__(x)
        assert isinstance(
            out, Variable
        ), f"Output must be a Variable but got {type(out)}"
        return out

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs
        for l in self.layers[:-1]:
            x = self.activation(l.apply(x))

        y = self.layers[-1].apply(x)
        return y
