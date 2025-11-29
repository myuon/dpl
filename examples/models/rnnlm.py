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
from dpl.layers import Layer
import dpl.layers as L


class RNNLM(Layer):
    """
    Recurrent Neural Network Language Model.

    Architecture:
        Input (word IDs) -> TimeEmbedding -> TimeRNN -> TimeAffine -> Output (scores)

    For training, use with TimeSoftmaxWithLoss to compute the loss.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of RNN hidden state
        stateful: If True, maintains state across batches (default: False)
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

        # Layers
        self.embed = L.TimeEmbedding(vocab_size, embedding_dim)
        self.rnn = L.TimeRNN(hidden_size, in_size=embedding_dim, stateful=stateful)
        self.affine = L.TimeAffine(vocab_size, in_size=hidden_size)

    def reset_state(self):
        """Reset RNN hidden state."""
        self.rnn.reset_state()

    def forward(self, *xs: Variable) -> Variable:
        """
        Forward pass of RNNLM.

        Args:
            xs: Tuple containing input word IDs with shape (batch_size, seq_len)

        Returns:
            Scores with shape (batch_size, seq_len, vocab_size)
        """
        (x,) = xs
        # x shape: (batch_size, seq_len)

        # Embedding: (batch_size, seq_len, embedding_dim)
        h = self.embed(x)

        # RNN: (batch_size, seq_len, hidden_size)
        h = self.rnn(h)

        # Affine: (batch_size, seq_len, vocab_size)
        scores = self.affine(h)

        return scores


class RNNLMWithLoss(Layer):
    """
    RNNLM with loss computation.

    This combines RNNLM with TimeSoftmaxWithLoss for training.

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
        self.rnnlm = RNNLM(vocab_size, embedding_dim, hidden_size, stateful=stateful)
        self.loss_layer = L.TimeSoftmaxWithLoss()

    def reset_state(self):
        """Reset RNN hidden state."""
        self.rnnlm.reset_state()

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

        # Get scores from RNNLM
        scores = self.rnnlm(inputs)

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
        return self.rnnlm(x)
