# %%
import dpl
import dpl.functions as F
import dpl.optimizers as O
import numpy as np
from datasets.ptb import load_data
from models.cbow import CBOWNegativeSamplingModel
from dpl.datasets import Dataset
from dpl.dataloaders import DataLoader


# %%
def create_contexts_target(corpus: np.ndarray, window_size: int = 1):
    """
    Create context-target pairs from corpus.

    Args:
        corpus: Array of word IDs
        window_size: Size of context window (default: 1)

    Returns:
        contexts: Array of context word IDs (num_samples, 2*window_size)
        target: Array of target word IDs (num_samples,)
    """
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts, dtype=np.int32), np.array(target, dtype=np.int32)


# %%
class CBOWDataset(Dataset):
    """
    Dataset for CBOW model that returns context words and target word.
    """

    def __init__(self, contexts: np.ndarray, target: np.ndarray):
        """
        Args:
            contexts: Array of context word IDs (num_samples, 2*window_size)
            target: Array of target word IDs (num_samples,)
        """
        self.contexts = contexts
        self.target = target

    def __getitem__(self, index):
        """
        Returns:
            contexts: Context word IDs for this sample (2,) for window_size=1
            target: Target word ID
        """
        return self.contexts[index], self.target[index]

    def __len__(self):
        return len(self.target)

    def prepare(self):
        pass


# %%
# Load PTB dataset
print("Loading PTB dataset...")
corpus, word_to_id, id_to_word = load_data("train")
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
max_epoch = 15
batch_size = 1000
lr = 0.01

print("\n" + "=" * 60)
print("Training Configuration")
print("=" * 60)
print(f"Vocabulary size: {vocabulary_size}")
print(f"Hidden size: {hidden_size}")
print(f"Negative samples: {sample_size}")
print(f"Max epoch: {max_epoch}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Training samples: {len(target)}")
print("=" * 60 + "\n")


# %%
# Create CBOW dataset and dataloader
cbow_dataset = CBOWDataset(contexts, target)
cbow_loader = DataLoader(cbow_dataset, batch_size=batch_size, shuffle=True)

# Create model with Negative Sampling
model = CBOWNegativeSamplingModel(
    vocabulary_size=vocabulary_size,
    hidden_size=hidden_size,
    sample_size=sample_size,
    corpus=corpus,  # Pass corpus for negative sampling
)

optimizer = O.Adam(lr=lr).setup(model)

# Custom training loop (Trainer doesn't support our use case well)
print("Starting training...")

from dpl.core import as_variable

loss_history = []

for epoch in range(1, max_epoch + 1):
    model.cleargrads()

    epoch_loss = 0.0
    batch_count = 0

    for contexts, target in cbow_loader:
        # Convert to Variables
        contexts_var = as_variable(contexts)
        target_var = as_variable(target)

        # Forward pass - model generates negative samples internally
        # Returns (target_score, negative_scores)
        target_score, negative_scores = model(contexts_var, target_var)

        # Compute loss
        loss = F.negative_sampling_loss(
            target_score, negative_scores, sample_size=sample_size
        )

        # Backward pass
        loss.backward()
        optimizer.update()
        model.cleargrads()

        # Track loss
        loss_data = loss.data
        assert loss_data is not None
        epoch_loss += float(loss_data) * len(target)  # type: ignore
        batch_count += len(target)

    avg_loss = epoch_loss / batch_count
    loss_history.append(avg_loss)
    print(f"Epoch {epoch}/{max_epoch} - Loss: {avg_loss:.4f}")

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)


# %%
# Plot training loss
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(range(1, max_epoch + 1), loss_history, linewidth=2, marker="o")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("CBOW with Negative Sampling - Training Loss (PTB)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nFinal loss: {loss_history[-1]:.4f}")


# %%
# Save the model
print("\nSaving model weights...")
model.save_weights("word2vec_ptb.npz")
print("Model saved to 'word2vec_ptb.npz'")


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
