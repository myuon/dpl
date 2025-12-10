"""
RNNLM (Recurrent Neural Network Language Model) implementation.

This model predicts the next word in a sequence using:
- Time Embedding: Convert word IDs to embeddings
- Time RNN: Process sequence with recurrent connections
- Time Affine: Project to vocabulary space
- Time Softmax with Loss: Compute loss
"""

import numpy as np
from dpl.core import Variable
from dpl.layers import Layer, StatefulLayer
from dpl.layers.layer import Parameter
import dpl.layers as L
import dpl.functions as F


class TimeAffineWithSharedWeight(Layer):
    """
    Time-distributed Affine layer that shares weights with an Embedding layer.

    This layer uses the transposed weight matrix from an embedding layer,
    implementing weight tying commonly used in language models.

    Args:
        shared_W: Parameter from embedding layer (vocab_size, embedding_dim)
        The layer will use shared_W.T as (embedding_dim, vocab_size) weight
    """
    def __init__(self, shared_W: Parameter):
        super().__init__()
        self.W = shared_W  # Share the parameter
        # Create bias parameter
        vocab_size = shared_W.shape[0]
        self.b = Parameter(np.zeros(vocab_size, dtype=np.float32), name="b")

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Tuple containing input with shape (batch_size, seq_len, input_dim)

        Returns:
            Output with shape (batch_size, seq_len, vocab_size)
        """
        (x,) = xs
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Reshape to (batch_size * seq_len, input_dim)
        x_reshaped = F.reshape(x, (batch_size * seq_len, -1))

        # Apply linear with transposed weight: x @ W.T + b
        # W shape: (vocab_size, embedding_dim), W.T: (embedding_dim, vocab_size)
        out = F.linear(x_reshaped, self.W.transpose(), self.b)
        # out shape: (batch_size * seq_len, vocab_size)

        # Reshape back to (batch_size, seq_len, vocab_size)
        out = F.reshape(out, (batch_size, seq_len, -1))

        return out


class RNNLMWithLoss(Layer):
    """
    RNNLM with loss computation.

    Architecture:
        Input (word IDs) -> TimeEmbedding -> TimeRNN -> TimeAffine -> TimeSoftmaxWithLoss -> Loss

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of RNN hidden state
        stateful: If True, maintains state across batches
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        stateful: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.stateful = stateful

        # Sequential model
        self.model = L.Sequential(
            L.TimeEmbedding(vocab_size, embedding_dim),
            L.TimeRNN(hidden_size, in_size=embedding_dim, stateful=stateful),
            L.TimeAffine(vocab_size, in_size=hidden_size),
        )
        self.loss_layer = L.TimeSoftmaxWithLoss()

    def reset_state(self):
        """Reset RNN hidden state."""
        self.model.reset_state()

    def forward(self, *xs: Variable) -> Variable:
        """
        Forward pass with loss computation.

        Args:
            xs: Tuple of (inputs, targets)
                - inputs: shape (batch_size, seq_len)
                - targets: shape (batch_size, seq_len)

        Returns:
            Loss value
        """
        inputs, targets = xs

        # Get scores from model
        scores = self.model(inputs)

        # Compute loss
        loss = self.loss_layer(scores, targets)

        return loss

    def predict(self, x: Variable) -> Variable:
        """
        Generate predictions without computing loss.

        Args:
            x: Input word IDs with shape (batch_size, seq_len)

        Returns:
            Scores with shape (batch_size, seq_len, vocab_size)
        """
        return self.model(x)


class BetterRNNLMWithLoss(Layer):
    """
    Better RNNLM with 2-layer LSTM, Dropout, Weight Tying, and loss computation.

    Uses 2-layer LSTM with Dropout for regularization and weight tying between
    embedding and output layers for better parameter efficiency.

    Architecture:
        Input (word IDs) -> TimeEmbedding -> Dropout -> TimeLSTM -> Dropout -> TimeLSTM -> Dropout -> TimeAffineWithSharedWeight -> TimeSoftmaxWithLoss -> Loss

    Weight Tying:
        The output layer uses the transposed embedding weight matrix, reducing
        parameters and improving performance.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings (must equal hidden_size for weight tying)
        hidden_size: Size of LSTM hidden state (must equal embedding_dim for weight tying)
        dropout_ratio: Dropout ratio for regularization (default: 0.5)
        stateful: If True, maintains state across batches
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        dropout_ratio: float = 0.5,
        stateful: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.stateful = stateful

        # Create embedding layer
        time_embed = L.TimeEmbedding(vocab_size, embedding_dim)

        # Create affine layer that shares weights with embedding layer
        # This implements weight tying: output layer uses transposed embedding weights
        time_affine = TimeAffineWithSharedWeight(time_embed.embed.W)

        # Sequential model with 2-layer LSTM, Dropout, and weight tying
        self.model = L.Sequential(
            time_embed,
            L.Dropout(dropout_ratio),
            L.TimeLSTM(hidden_size, in_size=embedding_dim, stateful=stateful),
            L.Dropout(dropout_ratio),
            L.TimeLSTM(hidden_size, in_size=hidden_size, stateful=stateful),
            L.Dropout(dropout_ratio),
            time_affine,
        )
        self.loss_layer = L.TimeSoftmaxWithLoss()

    def reset_state(self):
        """Reset LSTM hidden and cell state."""
        self.model.reset_state()

    def forward(self, *xs: Variable) -> Variable:
        """
        Forward pass with loss computation.

        Args:
            xs: Tuple of (inputs, targets)
                - inputs: shape (batch_size, seq_len)
                - targets: shape (batch_size, seq_len)

        Returns:
            Loss value
        """
        inputs, targets = xs

        # Get scores from model
        scores = self.model(inputs)

        # Compute loss
        loss = self.loss_layer(scores, targets)

        return loss

    def predict(self, x: Variable) -> Variable:
        """
        Generate predictions without computing loss.

        Args:
            x: Input word IDs with shape (batch_size, seq_len)

        Returns:
            Scores with shape (batch_size, seq_len, vocab_size)
        """
        return self.model(x)
