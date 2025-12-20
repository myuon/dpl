# %%
import dpl
import dpl.functions as F
import numpy as np
from models import CBOWModel
from utils.corpus import CBOWDataset, create_contexts_target, preprocess


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
optimizer = O.Adam(lr=lr).setup(model)

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
trainer.plot_history(history_types=["loss"], title="CBOW Training Loss")

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
