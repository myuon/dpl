# %%
import dpl
import dpl.functions as F
import dpl.optimizers as O
import numpy as np
from datasets.ptb import load_data
from models.cbow import CBOWNegativeSamplingModel
from dpl.dataloaders import DataLoader
from utils.corpus import CBOWDataset, create_contexts_target


# %%
# Load PTB dataset
corpus_size = 1000

print("Loading PTB dataset...")
corpus, word_to_id, id_to_word = load_data("train")
corpus = corpus[:corpus_size]
print(f"Corpus size: {len(corpus)}")  # type: ignore
print(f"Vocabulary size: {len(word_to_id)}")  # type: ignore

# Create context-target pairs
window_size = 5
print(f"\nCreating context-target pairs (window_size={window_size})...")
contexts, target = create_contexts_target(corpus, window_size=window_size)  # type: ignore
print(f"Contexts shape: {contexts.shape}")
print(f"Target shape: {target.shape}")

# Show some examples
print("\nFirst 5 context-target pairs:")
for i in range(min(5, len(target))):
    context_words = [id_to_word[c] for c in contexts[i]]  # type: ignore
    target_word = id_to_word[target[i]]  # type: ignore
    print(f"  Context: {context_words} -> Target: {target_word}")


# %%
# Hyperparameters
vocabulary_size = len(word_to_id)  # type: ignore
hidden_size = 100
sample_size = 5
max_epoch = 100
batch_size = 128

print("\n" + "=" * 60)
print("Training Configuration")
print("=" * 60)
print(f"Vocabulary size: {vocabulary_size}")
print(f"Hidden size: {hidden_size}")
print(f"Negative samples: {sample_size}")
print(f"Max epoch: {max_epoch}")
print(f"Batch size: {batch_size}")
print(f"Training samples: {len(target)}")
print("=" * 60 + "\n")


# %%
# Create CBOW dataset and dataloader
from dpl import Layer, Variable

cbow_dataset = CBOWDataset(contexts, target)
cbow_loader = DataLoader(cbow_dataset, batch_size=batch_size, shuffle=True)

# Create model with Negative Sampling
model = CBOWNegativeSamplingModel(
    vocabulary_size=vocabulary_size,
    hidden_size=hidden_size,
    sample_size=sample_size,
    corpus=corpus,  # Pass corpus for negative sampling
)

optimizer = O.Adam().setup(model)


# Create a wrapper class that adapts the model for Trainer
class Word2VecModelWrapper(Layer):
    """
    Wrapper to adapt CBOWNegativeSamplingModel for Trainer.

    Trainer expects model(x) to return a single value, but our model
    needs both contexts and target, and returns a tuple.
    This wrapper stores the target and handles the tuple return.
    """

    def __init__(self, model: CBOWNegativeSamplingModel):
        super().__init__()
        self.model = model
        self._target = None

    def forward(self, *xs: Variable):
        """
        Args:
            x: contexts from DataLoader

        Returns:
            Tuple of (target_score, negative_scores)
        """
        # x is contexts, self._target is set by preprocess_fn
        return self.model(*xs, self._target)

    def set_target(self, target):
        """Store target for next forward pass"""
        self._target = target


# Wrap the model
wrapped_model = Word2VecModelWrapper(model)


# Preprocess function that stores target in the wrapper
def preprocess_fn(x, t):
    """
    Store target in wrapper and return contexts.

    Args:
        x: contexts from DataLoader
        t: target from DataLoader

    Returns:
        (contexts, target) for Trainer
    """
    from dpl.core import as_variable

    wrapped_model.set_target(as_variable(t))
    return x, t


# Define loss function for Trainer
def word2vec_loss_fn(y, t):
    """
    Loss function for word2vec with negative sampling.

    Args:
        y: Tuple of (target_score, negative_scores) from model
        t: Target (unused, kept for Trainer interface compatibility)

    Returns:
        Loss value
    """
    target_score, negative_scores = y
    return F.negative_sampling_loss(
        target_score, negative_scores, sample_size=sample_size
    )


# Define perplexity metric
def perplexity_metric_fn(y, t):
    """
    Compute perplexity from the model output.

    Perplexity = exp(loss)

    Args:
        y: Tuple of (target_score, negative_scores) from model
        t: Target (unused, kept for Trainer interface compatibility)

    Returns:
        Perplexity value
    """
    target_score, negative_scores = y
    loss = F.negative_sampling_loss(
        target_score, negative_scores, sample_size=sample_size
    )
    # Perplexity = exp(loss)
    perplexity = F.exp(loss)
    return perplexity


# Train using Trainer
from dpl.trainer import Trainer

trainer = Trainer(
    model=wrapped_model,
    optimizer=optimizer,
    loss_fn=word2vec_loss_fn,
    metric_fn=perplexity_metric_fn,  # Add perplexity metric
    train_loader=cbow_loader,
    max_epoch=max_epoch,
    preprocess_fn=preprocess_fn,
)

trainer.run()

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)

# Get loss history from trainer
loss_history = trainer.train_loss_history


# %%
# Plot training loss
trainer.plot_history(
    history_type="loss",
    title="CBOW with Negative Sampling - Training Loss (PTB)",
)

print(f"\nFinal loss: {loss_history[-1]:.4f}")


# %%
# Plot perplexity over epochs
perplexity_history = trainer.train_metric_history

if len(perplexity_history) > 0:
    trainer.plot_history(
        history_type="metric",
        ylabel="Perplexity",
        title="CBOW with Negative Sampling - Perplexity (PTB)",
        color="green",
    )

    print(f"\nFinal perplexity: {perplexity_history[-1]:.4f}")


# %%
# Test word similarity using cosine similarity
def cos_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)


def most_similar(
    query: str, word_to_id: dict, id_to_word: dict, embeddings: np.ndarray, top: int = 5
):
    """
    Find most similar words to query word.

    Args:
        query: Query word
        word_to_id: Word to ID mapping
        id_to_word: ID to word mapping
        embeddings: Word embeddings matrix
        top: Number of top similar words to return
    """
    if query not in word_to_id:
        print(f"'{query}' is not in vocabulary")
        return

    query_id = word_to_id[query]
    query_vec = embeddings[query_id]

    # Compute similarity with all words
    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, embeddings[i])

    # Get top similar words (excluding the query word itself)
    top_indices = np.argsort(-similarity)[1 : top + 1]

    print(f"\n[Query] {query}")
    print("-" * 40)
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. {id_to_word[idx]}: {similarity[idx]:.4f}")


# %%
# Test word similarity
if model.cbow_ns.embed.W.data is not None:
    embeddings = model.cbow_ns.embed.W.data

    # Test some words
    test_words = ["you", "year", "car", "toyota"]

    for word in test_words:
        most_similar(word, word_to_id, id_to_word, embeddings, top=10)  # type: ignore
