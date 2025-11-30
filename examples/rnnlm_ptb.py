# %%
"""
RNNLM training on Penn Treebank dataset.

This script trains a Recurrent Neural Network Language Model (RNNLM) on the PTB corpus.
The model architecture is:
    Input (word IDs) -> TimeEmbedding -> TimeRNN -> TimeAffine -> TimeSoftmaxWithLoss
"""
import numpy as np
import dpl
import dpl.functions as F
import dpl.optimizers as O
from dpl.core import as_variable, Variable
from dpl.datasets import Dataset
from dpl.dataloaders import SequentialDataLoader
from dpl.trainer import Trainer
from dpl.layers import Layer
from datasets.ptb import load_data
from models.rnnlm import RNNLMWithLoss


# %%
# Simple dataset wrapper for corpus (sequence of word IDs)
class CorpusDataset(Dataset):
    """
    Dataset wrapper for corpus (sequence of word IDs).

    Each item returns a single word ID, with the dataset acting as a sliding window.
    """

    def __init__(self, corpus: np.ndarray):
        """
        Args:
            corpus: Array of word IDs
        """
        self.corpus = corpus

    def __getitem__(self, index):
        """
        Returns:
            word_id: Word ID at index (as single element for compatibility)
            word_id: Same word ID (target, though SequentialDataLoader handles targets differently)
        """
        return self.corpus[index], self.corpus[index]

    def __len__(self):
        return len(self.corpus)

    def prepare(self):
        pass


# %%
# Load PTB dataset
corpus_size = 1000

print("Loading PTB dataset...")
corpus, word_to_id, id_to_word = load_data("train")
corpus = corpus[:corpus_size]
corpus_val, _, _ = load_data("valid")
print(f"Training corpus size: {len(corpus)}")  # type: ignore
print(f"Validation corpus size: {len(corpus_val)}")  # type: ignore
print(f"Vocabulary size: {len(word_to_id)}")  # type: ignore

# Show some examples
print("\nFirst 20 words in corpus:")
print(" ".join([id_to_word[int(word_id)] for word_id in corpus[:20]]))  # type: ignore


# %%
# Hyperparameters
vocabulary_size = len(word_to_id)  # type: ignore
embedding_dim = 100
hidden_size = 100
max_epoch = 10
batch_size = 20
bptt_length = 35

print("\n" + "=" * 60)
print("Training Configuration")
print("=" * 60)
print(f"Vocabulary size: {vocabulary_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Hidden size: {hidden_size}")
print(f"Max epoch: {max_epoch}")
print(f"Batch size: {batch_size}")
print(f"BPTT length: {bptt_length}")
print(f"Training samples: {len(corpus)}")  # type: ignore
print("=" * 60 + "\n")


# %%
# Create dataset and dataloader
train_dataset = CorpusDataset(corpus)  # type: ignore
val_dataset = CorpusDataset(corpus_val)  # type: ignore

train_loader = SequentialDataLoader(
    train_dataset, batch_size=batch_size, bptt_length=bptt_length
)
val_loader = SequentialDataLoader(
    val_dataset, batch_size=batch_size, bptt_length=bptt_length
)

# Create model
rnnlm_model = RNNLMWithLoss(
    vocab_size=vocabulary_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    stateful=True,  # Maintain state across batches for better context
)

optimizer = O.Adam().setup(rnnlm_model)

print("Model created successfully!")
print(
    f"Model parameters: {sum(p.data.size for p in rnnlm_model.params() if p.data is not None)}"
)


# %%
# Create wrapper to adapt RNNLMWithLoss for Trainer
class RNNLMWrapper(Layer):
    """
    Wrapper to adapt RNNLMWithLoss for Trainer.

    Trainer expects model(x) to return a value, but RNNLMWithLoss
    needs both inputs and targets. This wrapper stores the target
    and returns the loss directly.
    """

    def __init__(self, model: RNNLMWithLoss):
        super().__init__()
        self.model = model
        self._target = None

    def forward(self, *xs: Variable):
        """
        Args:
            x: inputs from DataLoader

        Returns:
            Loss value
        """
        # x is inputs, self._target is set by preprocess_fn
        if self._target is None:
            raise ValueError("Target must be set before forward pass")
        return self.model(*xs, self._target)

    def set_target(self, target):
        """Store target for next forward pass"""
        self._target = target

    def reset_state(self):
        """Reset RNN hidden state."""
        self.model.reset_state()


# Wrap the model
model = RNNLMWrapper(rnnlm_model)


# Preprocess function that stores target in the wrapper
def preprocess_fn(x, t):
    """Store target in wrapper and return inputs."""
    model.set_target(as_variable(t))
    return x, t


# Callback for stateful model
def on_epoch_start(trainer: Trainer):
    """Reset model state at the start of each epoch."""
    # Reset model state
    if hasattr(trainer.model, "reset_state"):
        trainer.model.reset_state()  # type: ignore
    # Reset dataloaders
    if trainer.train_loader is not None and hasattr(trainer.train_loader, "reset"):
        trainer.train_loader.reset()
    if trainer.test_loader is not None and hasattr(trainer.test_loader, "reset"):
        trainer.test_loader.reset()


# %%
# Train using Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=lambda y, _t: y,  # y is already the loss from RNNLMWithLoss
    metric_fn=lambda y, _t: F.exp(y),  # Perplexity = exp(loss)
    train_loader=train_loader,
    test_loader=val_loader,
    max_epoch=max_epoch,
    on_epoch_start=on_epoch_start,
    preprocess_fn=preprocess_fn,
    truncate_bptt=True,  # Use truncated BPTT for memory efficiency
)

print("\nStarting training...")
trainer.run()

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)


# %%
# Plot training history
# Plot loss
trainer.plot_history(
    history_types=["loss", "test_loss"],
    title="RNNLM Training Loss (PTB)",
    figsize=(12, 6),
)

# Plot perplexity
trainer.plot_history(
    history_types=["metric", "test_metric"],
    ylabel="Perplexity",
    title="RNNLM Perplexity (PTB)",
    figsize=(12, 6),
)

# Print final results
train_loss_history = trainer.train_loss_history
test_loss_history = trainer.test_loss_history
train_ppl = trainer.train_metric_history
val_ppl = trainer.test_metric_history

print(f"\nFinal train loss: {train_loss_history[-1]:.4f} (PPL: {train_ppl[-1]:.2f})")
print(f"Final validation loss: {test_loss_history[-1]:.4f} (PPL: {val_ppl[-1]:.2f})")


# %%
# Save the model
print("\nSaving model weights...")
rnnlm_model.save_weights("rnnlm_ptb.npz")
print("Model saved to 'rnnlm_ptb.npz'")


# %%
# Text generation example
def generate_text(model, start_word, word_to_id, id_to_word, max_length=100):
    """
    Generate text starting from a given word.

    Args:
        model: Trained RNNLM model
        start_word: Starting word for generation
        word_to_id: Word to ID mapping
        id_to_word: ID to word mapping
        max_length: Maximum length of generated sequence

    Returns:
        Generated text as a string
    """
    if start_word not in word_to_id:
        print(f"'{start_word}' is not in vocabulary")
        return None

    rnnlm_model.reset_state()
    word_id = word_to_id[start_word]
    generated = [start_word]

    with dpl.no_grad():
        for _ in range(max_length):
            # Prepare input: shape (1, 1)
            x = as_variable(np.array([[word_id]], dtype=np.int32))

            # Get predictions (use predict method to get scores without loss)
            scores = rnnlm_model.predict(x)  # shape: (1, 1, vocab_size)

            # Get probabilities for next word
            # scores: (1, 1, vocab_size) -> (vocab_size,)
            assert scores.data is not None
            scores_flat = scores.data[0, 0, :]

            # Sample from probability distribution
            probs = np.exp(scores_flat) / np.sum(np.exp(scores_flat))
            word_id = np.random.choice(len(probs), p=probs)

            # Add to generated text
            word = id_to_word[int(word_id)]
            if word == "<eos>":
                break
            generated.append(word)

    return " ".join(generated)


# %%
# Generate some text
print("\n" + "=" * 60)
print("Text Generation Examples")
print("=" * 60)

start_words = ["the", "i", "we", "you"]
for start_word in start_words:
    text = generate_text(rnnlm_model, start_word, word_to_id, id_to_word, max_length=50)  # type: ignore
    if text:
        print(f"\nStarting with '{start_word}':")
        print(text)
