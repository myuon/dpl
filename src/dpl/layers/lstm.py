from dpl.core import Variable
from dpl.layers.layer import UnaryLayer
import dpl.layers as L
import dpl.functions as F


class LSTM(UnaryLayer):
    """
    Long Short-Term Memory (LSTM) layer.

    LSTM processes sequential data while maintaining both hidden state (h)
    and cell state (c), using three gates to control information flow:
    - Forget gate: Controls what information to discard from cell state
    - Input gate: Controls what new information to store in cell state
    - Output gate: Controls what information to output from cell state
    """

    def __init__(self, hidden_size: int, in_size=None):
        super().__init__()
        self.hidden_size = hidden_size

        # Input gate
        self.x2i = L.Linear(hidden_size, in_size=in_size)
        self.h2i = L.Linear(hidden_size, in_size=hidden_size, no_bias=True)

        # Forget gate
        self.x2f = L.Linear(hidden_size, in_size=in_size)
        self.h2f = L.Linear(hidden_size, in_size=hidden_size, no_bias=True)

        # Output gate
        self.x2o = L.Linear(hidden_size, in_size=in_size)
        self.h2o = L.Linear(hidden_size, in_size=hidden_size, no_bias=True)

        # Cell state candidate
        self.x2c = L.Linear(hidden_size, in_size=in_size)
        self.h2c = L.Linear(hidden_size, in_size=hidden_size, no_bias=True)

        # Hidden state and cell state
        self.h = None
        self.c = None

    def reset_state(self):
        """Reset both hidden state and cell state."""
        self.h = None
        self.c = None

    def forward(self, *xs: Variable) -> Variable:
        """
        Forward pass through LSTM.

        Args:
            xs: Input variable (usually a single input)

        Returns:
            Hidden state variable
        """
        x = xs[0]

        if self.h is None:
            # Initialize hidden state and cell state with zeros on first call
            # Input gate
            i = F.sigmoid(self.x2i(x))
            # Forget gate
            f = F.sigmoid(self.x2f(x))
            # Output gate
            o = F.sigmoid(self.x2o(x))
            # Cell state candidate
            c_hat = F.tanh(self.x2c(x))

            # New cell state
            c_new = i * c_hat
            # New hidden state
            h_new = o * F.tanh(c_new)
        else:
            # Input gate
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            # Forget gate
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            # Output gate
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            # Cell state candidate
            c_hat = F.tanh(self.x2c(x) + self.h2c(self.h))

            # New cell state: forget old state and add new candidate
            c_new = f * self.c + i * c_hat
            # New hidden state
            h_new = o * F.tanh(c_new)

        self.h = h_new
        self.c = c_new

        return h_new
