import numpy as np
from dpl.core import Variable
from dpl.layers.layer import Parameter, UnaryLayer
import dpl.functions as F


class Embedding(UnaryLayer):
    """
    Embedding layer that converts word IDs to dense vectors.

    Args:
        vocab_size: Size of the vocabulary (number of unique words/tokens)
        embedding_dim: Dimension of the embedding vectors
        W: Optional pre-initialized embedding matrix. If None, will be initialized randomly.
        dtype: Data type for the embedding matrix
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        W: np.ndarray | None = None,
        dtype=np.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        if W is None:
            # Initialize embedding matrix with random values
            W = np.random.randn(vocab_size, embedding_dim).astype(dtype) * np.sqrt(
                1 / embedding_dim
            )

        self.W = Parameter(W, name="W")
        self.idx = None

    def forward(self, *xs: Variable) -> Variable:
        """
        Forward pass of the embedding layer.

        Args:
            xs: Tuple containing a single Variable with word IDs (shape: (batch_size,) or (batch_size, seq_len))

        Returns:
            Embedded vectors (shape: (batch_size, embedding_dim) or (batch_size, seq_len, embedding_dim))
        """
        (idx,) = xs
        self.idx = idx

        # Use the embed function which supports automatic differentiation
        return F.embed(idx, self.W)
