import dpl
import dpl.functions as F
from dpl import Layer, Model, Parameter, Variable, as_variable
import numpy as np


class CBOWLayer(Layer):
    """
    Simple CBOW (Continuous Bag of Words) model

    Architecture:
    - Input: 2 context word vectors
    - Win: Embedding layer (vocabulary_size x hidden_size)
    - Average the two embedded vectors (multiply by 0.5)
    - Wout: Output layer (hidden_size x vocabulary_size)
    - Softmax cross entropy loss
    """

    def __init__(self, vocabulary_size: int, hidden_size: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size

        # Win: Input embedding matrix
        Win_data = (
            np.random.randn(vocabulary_size, hidden_size).astype(np.float32) * 0.01
        )
        self.Win = Parameter(Win_data, name="Win")

        # Wout: Output weight matrix
        Wout_data = (
            np.random.randn(hidden_size, vocabulary_size).astype(np.float32) * 0.01
        )
        self.Wout = Parameter(Wout_data, name="Wout")

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            context0: First context word (batch_size,) - word indices
            context1: Second context word (batch_size,) - word indices

        Returns:
            Logits for target word prediction (batch_size, vocabulary_size)
        """
        context0, context1 = xs

        # Embed context words using Win
        # Use indexing to select rows from Win matrix based on word indices
        h0 = self.Win[context0.data_required]  # type: ignore  # (batch_size, hidden_size)
        h1 = self.Win[context1.data_required]  # type: ignore  # (batch_size, hidden_size)

        # Average the two context vectors
        h = (h0 + h1) * 0.5  # (batch_size, hidden_size)

        # Apply output layer
        out = F.matmul(h, self.Wout)  # (batch_size, vocabulary_size)

        return out


class CBOWModel(Model):
    """
    Wrapper for SimpleCBOW that takes contexts as a single input.
    This is compatible with the Trainer API.
    """

    def __init__(self, vocabulary_size: int, hidden_size: int):
        super().__init__()
        self.cbow = CBOWLayer(vocabulary_size=vocabulary_size, hidden_size=hidden_size)

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            contexts: Context words (batch_size, 2) - contains [context0, context1]

        Returns:
            Logits for target word prediction (batch_size, vocabulary_size)
        """
        (contexts,) = xs
        # Split contexts into context0 and context1
        context0 = contexts[:, 0]
        context1 = contexts[:, 1]

        # Convert to Variables
        context0_var = as_variable(context0)
        context1_var = as_variable(context1)

        # Forward through CBOW model
        return self.cbow(context0_var, context1_var)
