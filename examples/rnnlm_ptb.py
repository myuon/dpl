# %%
"""
RNNLM training on Penn Treebank dataset.

This script trains a Recurrent Neural Network Language Model (RNNLM) on the PTB corpus.
The model architecture is:
    Input (word IDs) -> TimeEmbedding -> TimeRNN -> TimeAffine -> TimeSoftmaxWithLoss
"""
import numpy as np
import dpl
import dpl.optimizers as O
from dpl.core import as_variable
from dpl.datasets import Dataset
from dpl.dataloaders import SequentialDataLoader
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
print("Loading PTB dataset...")
corpus, word_to_id, id_to_word = load_data("train")
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
bptt_length = 35  # Backpropagation Through Time length

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
model = RNNLMWithLoss(
    vocab_size=vocabulary_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    stateful=True,  # Maintain state across batches for better context
)

optimizer = O.SGD(lr=1.0).setup(model)

print("Model created successfully!")
print(f"Model parameters: {sum(p.data.size for p in model.params() if p.data is not None)}")


# %%
# Training loop
print("\nStarting training...")
train_loss_history = []
val_loss_history = []

for epoch in range(1, max_epoch + 1):
    # Training
    model.reset_state()
    train_loader.reset()
    epoch_loss = 0.0
    batch_count = 0

    for xs_batch, ts_batch in train_loader:
        # xs_batch shape: (batch_size, bptt_length)
        # ts_batch shape: (batch_size, bptt_length)

        # Convert to Variables
        xs_var = as_variable(xs_batch)
        ts_var = as_variable(ts_batch)

        # Forward pass
        loss = model(xs_var, ts_var)

        # Backward pass
        model.cleargrads()
        loss.backward()
        loss.unchain_backward()  # Truncate computational graph for memory efficiency
        optimizer.update()

        # Track loss
        loss_data = loss.data
        assert loss_data is not None
        epoch_loss += float(loss_data)
        batch_count += 1

    avg_train_loss = epoch_loss / batch_count
    train_loss_history.append(avg_train_loss)

    # Validation
    model.reset_state()
    val_loader.reset()
    val_loss = 0.0
    val_batch_count = 0

    with dpl.no_grad():
        for xs_batch, ts_batch in val_loader:
            xs_var = as_variable(xs_batch)
            ts_var = as_variable(ts_batch)

            loss = model(xs_var, ts_var)

            loss_data = loss.data
            assert loss_data is not None
            val_loss += float(loss_data)
            val_batch_count += 1

    avg_val_loss = val_loss / val_batch_count
    val_loss_history.append(avg_val_loss)

    # Print progress
    train_ppl = np.exp(avg_train_loss)
    val_ppl = np.exp(avg_val_loss)
    print(f"Epoch {epoch}/{max_epoch} - "
          f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) - "
          f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})")

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)


# %%
# Plot training history
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(range(1, max_epoch + 1), train_loss_history, linewidth=2, marker="o", label="Train")
ax1.plot(range(1, max_epoch + 1), val_loss_history, linewidth=2, marker="s", label="Validation")
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("RNNLM Training Loss (PTB)", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Perplexity plot
train_ppl = [np.exp(loss) for loss in train_loss_history]
val_ppl = [np.exp(loss) for loss in val_loss_history]
ax2.plot(range(1, max_epoch + 1), train_ppl, linewidth=2, marker="o", label="Train")
ax2.plot(range(1, max_epoch + 1), val_ppl, linewidth=2, marker="s", label="Validation")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Perplexity", fontsize=12)
ax2.set_title("RNNLM Perplexity (PTB)", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal train loss: {train_loss_history[-1]:.4f} (PPL: {np.exp(train_loss_history[-1]):.2f})")
print(f"Final validation loss: {val_loss_history[-1]:.4f} (PPL: {np.exp(val_loss_history[-1]):.2f})")


# %%
# Save the model
print("\nSaving model weights...")
model.save_weights("rnnlm_ptb.npz")
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

    model.reset_state()
    word_id = word_to_id[start_word]
    generated = [start_word]

    with dpl.no_grad():
        for _ in range(max_length):
            # Prepare input: shape (1, 1)
            x = as_variable(np.array([[word_id]], dtype=np.int32))

            # Get predictions
            scores = model.rnnlm(x)  # shape: (1, 1, vocab_size)

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
    text = generate_text(model, start_word, word_to_id, id_to_word, max_length=50)  # type: ignore
    if text:
        print(f"\nStarting with '{start_word}':")
        print(text)
