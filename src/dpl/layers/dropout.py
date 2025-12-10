from dpl.core import Variable
from dpl.layers.layer import Layer
import dpl.functions as F


class Dropout(Layer):
    """
    Dropout layer for regularization.

    During training, randomly sets input elements to 0 with probability dropout_ratio,
    and scales the remaining elements by 1/(1-dropout_ratio).
    During testing, returns the input unchanged.

    Args:
        dropout_ratio: Probability of dropping out each element (default: 0.5)
    """

    def __init__(self, dropout_ratio: float = 0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio

    def forward(self, *xs: Variable) -> Variable:
        return F.dropout(xs[0], dropout_ratio=self.dropout_ratio)
