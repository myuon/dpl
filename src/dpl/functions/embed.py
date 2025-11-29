from dpl.core import Variable, BinaryFunction, ndarray, get_array_module


class EmbedID(BinaryFunction):
    """
    Embedding function that extracts rows from W based on indices in idx.

    Forward: out = W[idx]
    Backward: gradient is accumulated at the indexed positions
    """
    def forward(self, *xs: ndarray) -> ndarray:
        W, idx = xs
        self.idx = idx
        # Extract rows from W using indices in idx
        out = W[idx]
        return out

    def backward(self, *gys: Variable) -> tuple[Variable, Variable]:
        idx = self.idx
        (gy,) = gys
        W, _ = self.inputs  # W and idx are the inputs

        # Create gradient with same shape as W
        xp = get_array_module(gy.data)
        gW = xp.zeros_like(W.data)

        # Accumulate gradients at the indexed positions
        # add.at is used to handle duplicate indices correctly
        xp.add.at(gW, idx, gy.data)  # type: ignore

        # Return gradient for W and None for idx (indices don't need gradients)
        return Variable(gW), Variable(xp.zeros_like(idx))


def embed(idx: Variable | ndarray, W: Variable | ndarray) -> Variable:
    """
    Embedding function that extracts rows from W based on indices in idx.

    This is essentially a lookup operation: W[idx]

    Args:
        idx: Indices (word IDs) to extract. Shape: (batch_size,) or (batch_size, seq_len)
        W: Embedding matrix. Shape: (vocab_size, embedding_dim)

    Returns:
        Embedded vectors. Shape: (batch_size, embedding_dim) or (batch_size, seq_len, embedding_dim)
    """
    return EmbedID().apply(W, idx)
