"""
Time-distributed layers for sequence processing.

These layers apply the same operation at each time step in a sequence.
"""
import numpy as np
from dpl.core import Variable
from dpl.layers.layer import Layer
from dpl.layers.embedding import Embedding
from dpl.layers.linear import Linear
from dpl.layers.rnn import RNN
from dpl.layers.lstm import LSTM
import dpl.functions as F


class TimeEmbedding(Layer):
    """
    Time-distributed Embedding layer.

    Applies embedding lookup at each time step of a sequence.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embedding vectors
        W: Optional pre-initialized embedding matrix
    """
    def __init__(self, vocab_size: int, embedding_dim: int, W: np.ndarray | None = None):
        super().__init__()
        self.embed = Embedding(vocab_size, embedding_dim, W=W)

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Tuple containing input with shape (batch_size, seq_len)

        Returns:
            Embedded sequence with shape (batch_size, seq_len, embedding_dim)
        """
        (x,) = xs
        # x shape: (batch_size, seq_len)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Apply embedding - F.embed handles multi-dimensional input
        out = self.embed(x)
        # out shape: (batch_size, seq_len, embedding_dim)

        return out


class TimeRNN(Layer):
    """
    Time-distributed RNN layer.

    Processes a sequence using an RNN, maintaining hidden state across time steps.

    Args:
        hidden_size: Size of hidden state
        in_size: Size of input features (optional, inferred from first input)
        stateful: If True, maintains state across batches
    """
    def __init__(self, hidden_size: int, in_size: int | None = None, stateful: bool = False):
        super().__init__()
        self.rnn = RNN(hidden_size, in_size=in_size)
        self.stateful = stateful

    def reset_state(self):
        """Reset the hidden state."""
        self.rnn.reset_state()

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Tuple containing input with shape (batch_size, seq_len, input_dim)

        Returns:
            Hidden states at each time step with shape (batch_size, seq_len, hidden_size)
        """
        (x,) = xs
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if not self.stateful:
            self.reset_state()

        # Process each time step and concatenate outputs
        hs = []
        for t in range(seq_len):
            # Get input at time t: (batch_size, input_dim)
            x_t = x[:, t, :]
            h = self.rnn(x_t)
            # Reshape to (batch_size, 1, hidden_size) for concatenation
            h_reshaped = F.reshape(h, (batch_size, 1, -1))
            hs.append(h_reshaped)

        # Concatenate along sequence dimension
        # Result: (batch_size, seq_len, hidden_size)
        out = F.concat(hs, axis=1)

        return out


class TimeLSTM(Layer):
    """
    Time-distributed LSTM layer.

    Processes a sequence using an LSTM, maintaining hidden and cell state across time steps.

    Args:
        hidden_size: Size of hidden state
        in_size: Size of input features (optional, inferred from first input)
        stateful: If True, maintains state across batches
    """
    def __init__(self, hidden_size: int, in_size: int | None = None, stateful: bool = False):
        super().__init__()
        self.lstm = LSTM(hidden_size, in_size=in_size)
        self.stateful = stateful

    def reset_state(self):
        """Reset the hidden and cell state."""
        self.lstm.reset_state()

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Tuple containing input with shape (batch_size, seq_len, input_dim)

        Returns:
            Hidden states at each time step with shape (batch_size, seq_len, hidden_size)
        """
        (x,) = xs
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if not self.stateful:
            self.reset_state()

        # Process each time step and concatenate outputs
        hs = []
        for t in range(seq_len):
            # Get input at time t: (batch_size, input_dim)
            x_t = x[:, t, :]
            h = self.lstm(x_t)
            # Reshape to (batch_size, 1, hidden_size) for concatenation
            h_reshaped = F.reshape(h, (batch_size, 1, -1))
            hs.append(h_reshaped)

        # Concatenate along sequence dimension
        # Result: (batch_size, seq_len, hidden_size)
        out = F.concat(hs, axis=1)

        return out


class TimeAffine(Layer):
    """
    Time-distributed Affine (Linear) layer.

    Applies the same linear transformation at each time step.

    Args:
        out_size: Output dimension
        in_size: Input dimension (optional, inferred from first input)
        no_bias: If True, no bias term
    """
    def __init__(self, out_size: int, in_size: int | None = None, no_bias: bool = False):
        super().__init__()
        self.linear = Linear(out_size, in_size=in_size, no_bias=no_bias)

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Tuple containing input with shape (batch_size, seq_len, input_dim)

        Returns:
            Output with shape (batch_size, seq_len, out_size)
        """
        (x,) = xs
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Reshape to (batch_size * seq_len, input_dim)
        x_reshaped = F.reshape(x, (batch_size * seq_len, -1))

        # Apply linear layer
        out = self.linear(x_reshaped)
        # out shape: (batch_size * seq_len, out_size)

        # Reshape back to (batch_size, seq_len, out_size)
        out = F.reshape(out, (batch_size, seq_len, -1))

        return out


class TimeSoftmaxWithLoss(Layer):
    """
    Time-distributed Softmax with Cross Entropy Loss.

    Computes softmax cross entropy loss at each time step and averages.

    This is commonly used as the output layer for sequence models like RNNLMs.
    """
    def __init__(self):
        super().__init__()
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Tuple of (scores, targets)
                - scores: shape (batch_size, seq_len, vocab_size)
                - targets: shape (batch_size, seq_len)

        Returns:
            Average loss across all time steps
        """
        scores, targets = xs
        # scores shape: (batch_size, seq_len, vocab_size)
        # targets shape: (batch_size, seq_len)

        batch_size = scores.shape[0]
        seq_len = scores.shape[1]

        # Reshape for softmax cross entropy
        # scores: (batch_size * seq_len, vocab_size)
        # targets: (batch_size * seq_len,)
        scores_reshaped = F.reshape(scores, (batch_size * seq_len, -1))
        targets_reshaped = F.reshape(targets, (batch_size * seq_len,))

        # Compute softmax cross entropy loss
        loss = F.softmax_cross_entropy(scores_reshaped, targets_reshaped)

        self.loss = loss
        self.y = scores
        self.t = targets

        return loss
