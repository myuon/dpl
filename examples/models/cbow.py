import dpl
import dpl.functions as F
from dpl import Layer, Model, Parameter, Variable, as_variable
from dpl.layers import Embedding
import numpy as np


class UnigramSampler:
    """
    Unigram sampler for negative sampling.
    Samples words according to their frequency raised to the power of 0.75.
    """

    def __init__(self, corpus: np.ndarray, power: float = 0.75, sample_size: int = 5):
        """
        Args:
            corpus: Word ID corpus
            power: Power to raise word counts (typically 0.75)
            sample_size: Number of negative samples per positive sample
        """
        self.sample_size = sample_size

        # Count word frequencies
        counts = np.bincount(corpus)
        vocab_size = len(counts)

        # Compute sampling probabilities (word_count^power)
        self.probabilities = np.power(counts, power).astype(np.float64)
        self.probabilities /= np.sum(self.probabilities)

        self.vocab_size = vocab_size

    def sample(self, target: np.ndarray) -> np.ndarray:
        """
        Sample negative samples for each target word.

        Args:
            target: Target word IDs (batch_size,)

        Returns:
            Negative samples (batch_size, sample_size)
        """
        batch_size = target.shape[0] if target.ndim > 0 else 1

        # Sample from the distribution
        negative_samples = np.random.choice(
            self.vocab_size,
            size=(batch_size, self.sample_size),
            replace=True,
            p=self.probabilities,
        )

        return negative_samples


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


class CBOWNegativeSamplingLayer(Layer):
    """
    CBOW with Negative Sampling for efficient training.

    Instead of computing softmax over the entire vocabulary, this uses
    negative sampling to only compute loss for the target word and
    a small number of negative samples.
    """

    def __init__(self, vocabulary_size: int, hidden_size: int, sample_size: int = 5):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.sample_size = sample_size

        # Embedding layer for input
        self.embed = Embedding(vocabulary_size, hidden_size)

        # Output embedding layer (used for computing scores)
        self.embed_out = Embedding(vocabulary_size, hidden_size)

    def __call__(self, *inputs: Variable):  # type: ignore
        """Override to handle tuple return"""
        import weakref

        output = self.forward(*inputs)

        self.inputs = [weakref.ref(x) for x in inputs]
        # Don't store outputs since we return a tuple
        return output

    def forward(self, *xs: Variable) -> tuple[Variable, Variable]:  # type: ignore
        """
        Args:
            context0: First context word (batch_size,) - word indices
            context1: Second context word (batch_size,) - word indices
            target: Target word (batch_size,) - word indices
            negative_samples: Negative samples (batch_size, sample_size) - word indices

        Returns:
            Tuple of (target_score, negative_scores):
            - target_score: (batch_size, 1)
            - negative_scores: (batch_size, sample_size)
        """
        context0, context1, target, negative_samples = xs

        # Embed context words
        h0 = self.embed(context0)  # (batch_size, hidden_size)
        h1 = self.embed(context1)  # (batch_size, hidden_size)

        # Average the context vectors
        h = (h0 + h1) * 0.5  # (batch_size, hidden_size)

        # Get target embedding
        target_embed = self.embed_out(target)  # (batch_size, hidden_size)

        # Compute target score (dot product)
        target_score = F.sum(h * target_embed, axis=1, keepdims=True)  # (batch_size, 1)

        # Get negative embeddings and compute scores
        # We compute scores individually for each column to preserve computation graph
        # negative_samples: (batch_size, sample_size)
        neg_data = negative_samples.data_required
        batch_size = neg_data.shape[0]

        # Compute scores for all negative samples
        negative_scores_list = []
        for i in range(self.sample_size):
            neg_sample_i = negative_samples[:, i]  # (batch_size,)
            neg_embed_i = self.embed_out(neg_sample_i)  # (batch_size, hidden_size)
            score_i = F.sum(h * neg_embed_i, axis=1, keepdims=True)  # (batch_size, 1)
            negative_scores_list.append(score_i)

        # Concatenate all negative scores along axis=1
        # Each score_i is (batch_size, 1), result is (batch_size, sample_size)
        negative_scores = F.concat(negative_scores_list, axis=1)

        # Return both scores separately to preserve computation graph
        return target_score, negative_scores


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


class CBOWNegativeSamplingModel(Model):
    """
    CBOW model with Negative Sampling.
    This model is more efficient for large vocabularies.
    """

    def __init__(
        self,
        vocabulary_size: int,
        hidden_size: int,
        sample_size: int = 5,
        corpus: np.ndarray | None = None,
    ):
        super().__init__()
        self.cbow_ns = CBOWNegativeSamplingLayer(
            vocabulary_size=vocabulary_size,
            hidden_size=hidden_size,
            sample_size=sample_size,
        )
        self.sample_size = sample_size

        # Initialize sampler if corpus is provided
        self.sampler = None
        if corpus is not None:
            self.sampler = UnigramSampler(corpus, sample_size=sample_size)

    def __call__(self, *inputs: Variable):  # type: ignore
        """Override to handle tuple return"""
        import weakref

        output = self.forward(*inputs)

        self.inputs = [weakref.ref(x) for x in inputs]
        # Don't store outputs since we return a tuple
        return output

    def forward(self, *xs: Variable) -> tuple[Variable, Variable]:  # type: ignore
        """
        Args:
            contexts: Context words (batch_size, 2) - contains [context0, context1]
            target: Target words (batch_size,)
            negative_samples: Optional negative samples (batch_size, sample_size)
                             If not provided, will use sampler to generate

        Returns:
            Tuple of (target_score, negative_scores):
            - target_score: (batch_size, 1)
            - negative_scores: (batch_size, sample_size)
        """
        if len(xs) == 2:
            contexts, target = xs
            # Generate negative samples using sampler
            if self.sampler is None:
                raise ValueError(
                    "Sampler not initialized. Provide corpus or negative_samples."
                )
            target_data = target.data
            assert target_data is not None
            negative_samples_data = self.sampler.sample(target_data)  # type: ignore
            negative_samples = as_variable(negative_samples_data)
        else:
            contexts, target, negative_samples = xs

        # Split contexts into context0 and context1
        context0 = contexts[:, 0]
        context1 = contexts[:, 1]

        # Convert to Variables
        context0_var = as_variable(context0)
        context1_var = as_variable(context1)

        # Forward through CBOW with negative sampling
        return self.cbow_ns(context0_var, context1_var, target, negative_samples)
