# %%
import dpl
import dpl.functions as F
from dpl import Layer, Parameter, Variable, as_variable
import numpy as np


class SimpleCBOW(Layer):
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


# %%
def preprocess(text: str):
    """
    Preprocess text corpus and create word-to-id and id-to-word mappings.

    Args:
        text: Input text string

    Returns:
        corpus: List of word IDs
        word_to_id: Dictionary mapping words to IDs
        id_to_word: Dictionary mapping IDs to words
    """
    # Convert to lowercase and split into words
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split()

    # Create word-to-id and id-to-word mappings
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # Create corpus (list of word IDs)
    corpus = np.array([word_to_id[word] for word in words], dtype=np.int32)

    return corpus, word_to_id, id_to_word


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
# Example: Preprocess text
text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

print(f"Corpus: {corpus}")
print(f"Word to ID: {word_to_id}")
print(f"ID to Word: {id_to_word}")
print(f"Vocabulary size: {len(word_to_id)}")

# Create context-target pairs
contexts, target = create_contexts_target(corpus, window_size=1)
print(f"\nContexts shape: {contexts.shape}")
print(f"Target shape: {target.shape}")
print(f"\nFirst 3 context-target pairs:")
for i in range(min(3, len(target))):
    context_words = [id_to_word[c] for c in contexts[i]]
    target_word = id_to_word[target[i]]
    print(f"  Context: {context_words} -> Target: {target_word}")


# %%
# CBOW Dataset
from dpl.datasets import Dataset


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
# CBOW Model wrapper for Trainer
from dpl import Model


class CBOWModel(Model):
    """
    Wrapper for SimpleCBOW that takes contexts as a single input.
    This is compatible with the Trainer API.
    """

    def __init__(self, vocabulary_size: int, hidden_size: int):
        super().__init__()
        self.cbow = SimpleCBOW(vocabulary_size=vocabulary_size, hidden_size=hidden_size)

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


# %%
# Training with Trainer
import dpl.optimizers as O
import matplotlib.pyplot as plt
from dpl.dataloaders import DataLoader
from dpl import Trainer

# Hyperparameters
vocabulary_size = len(word_to_id)
hidden_size = 5
max_epoch = 100
batch_size = 2
lr = 1.0

# Create CBOW dataset and dataloader
cbow_dataset = CBOWDataset(contexts, target)
cbow_loader = DataLoader(cbow_dataset, batch_size=batch_size, shuffle=True)

# Create model
model = CBOWModel(vocabulary_size=vocabulary_size, hidden_size=hidden_size)
optimizer = O.SGD(lr=lr).setup(model)

# Create Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=F.softmax_cross_entropy,
    train_loader=cbow_loader,
    max_epoch=max_epoch,
)

print(f"\nTraining with {len(target)} samples")
print(f"Vocabulary size: {vocabulary_size}")
print(f"Hidden size: {hidden_size}")
print(f"Max epoch: {max_epoch}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}\n")

# Run training
trainer.run()

# Get loss history from trainer
loss_history = trainer.train_loss_history

print("\nTraining completed!")

# %%
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_epoch + 1), loss_history, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CBOW Training Loss")
plt.grid(True)
plt.show()

# %%
# Test the learned embeddings
print("\nLearned word embeddings (Win):")
if model.cbow.Win.data is not None:
    for word_id, word in id_to_word.items():
        embedding = model.cbow.Win.data[word_id]
        print(f"{word}: {embedding}")

# %%
# Visualize word embeddings as heatmap
if model.cbow.Win.data is not None:
    # Get embedding matrix
    embeddings = model.cbow.Win.data  # Shape: (vocabulary_size, hidden_size)

    # Create word labels in order
    word_labels = [id_to_word[i] for i in range(len(id_to_word))]

    # Create dimension labels
    dim_labels = [f"Dim {i}" for i in range(hidden_size)]

    # Create heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(embeddings, cmap="coolwarm", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(hidden_size))
    ax.set_yticks(np.arange(vocabulary_size))
    ax.set_xticklabels(dim_labels)
    ax.set_yticklabels(word_labels)

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Embedding Value", rotation=270, labelpad=20)

    # Add text annotations
    for i in range(vocabulary_size):
        for j in range(hidden_size):
            text = ax.text(
                j,
                i,
                f"{embeddings[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if abs(embeddings[i, j]) < 1.5 else "white",
                fontsize=9,
            )

    ax.set_title("Word Embeddings Heatmap", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Embedding Dimensions", fontsize=12)
    ax.set_ylabel("Words", fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
