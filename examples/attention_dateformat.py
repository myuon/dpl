"""
Attention Seq2Seq training script for date format conversion.

This script trains an Attention Seq2Seq model to convert dates:
    Input: "september 27, 1994" -> Output: "1994-09-27"
"""

import numpy as np
import dpl
import dpl.optimizers as O
from dpl.core import Variable, as_variable
from dpl.dataloaders import DataLoader
from dpl.trainer import Trainer
from dpl.layers import Layer
import datasets
from models import AttentionSeq2SeqWithLoss


# %%
# Load dataset
print("Loading DateFormat dataset...")
train_set = datasets.DateFormat(
    num_samples=50000, train=True, seed=42, reverse_input=True
)
test_set = datasets.DateFormat(
    num_samples=1000, train=False, seed=42, reverse_input=True
)

print(f"Training samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")
print(f"Vocabulary size: {datasets.DateFormat.VOCAB_SIZE}")
print(f"Input length: {train_set.input_len}")
print(f"Output length: {train_set.output_len}")

# Show examples
print("\nExamples:")
for i in range(min(3, len(train_set))):
    x, t = train_set[i]
    print(
        f"  Input: '{datasets.DateFormat.decode(x[::-1])}' -> Target: '{datasets.DateFormat.decode(t)}'"
    )


# %%
# Hyperparameters
vocab_size = datasets.DateFormat.VOCAB_SIZE
embedding_dim = 16
hidden_size = 128
max_epoch = 5
batch_size = 128
lr = 0.01
max_grad = 5.0

print("=" * 60)
print("Training Configuration")
print("=" * 60)
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Hidden size: {hidden_size}")
print(f"Max epoch: {max_epoch}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print("=" * 60)


# %%
# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# %%
# Create model
model = AttentionSeq2SeqWithLoss(vocab_size, embedding_dim, hidden_size)

print("Model created successfully!")
print(
    f"Model parameters: {sum(p.data.size for p in model.params() if p.data is not None)}"
)


# %%
# Wrapper to adapt AttentionSeq2SeqWithLoss for Trainer
class AttentionSeq2SeqWrapper(Layer):
    """Wrapper to adapt AttentionSeq2SeqWithLoss for Trainer."""

    def __init__(self, model: AttentionSeq2SeqWithLoss, start_id: int, output_len: int):
        super().__init__()
        self.model = model
        self.start_id = start_id
        self.output_len = output_len
        self._input = None
        self._decoder_input = None
        self._target = None
        self._target_np = None

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs
        if self._decoder_input is None or self._target is None:
            raise ValueError("decoder_input and target must be set before forward")
        self._input = x
        return self.model(x, self._decoder_input, self._target)

    def set_inputs(self, decoder_input, target, target_np):
        """Store decoder_input and target for next forward pass."""
        self._decoder_input = decoder_input
        self._target = target
        self._target_np = target_np

    def generate(self, input_seq, start_id, max_len):
        """Generate output sequence."""
        return self.model.generate(input_seq, start_id, max_len)

    def compute_accuracy(self) -> float:
        """Compute sequence-level accuracy."""
        if self._input is None or self._target_np is None:
            return 0.0
        generated = self.model.generate(self._input, self.start_id, self.output_len)
        correct = np.sum(np.all(generated == self._target_np, axis=1))
        return correct / len(self._target_np)


# %%
# Wrap the model
start_id = datasets.DateFormat.START_ID
output_len = train_set.output_len
wrapped_model = AttentionSeq2SeqWrapper(model, start_id, output_len)
optimizer = O.Adam(lr=lr).setup(wrapped_model)


def preprocess_fn(x, t):
    """Create decoder input and store in wrapper."""
    batch_size = x.shape[0]
    start_tokens = np.full((batch_size, 1), start_id, dtype=np.int32)
    decoder_input = np.concatenate([start_tokens, t[:, :-1]], axis=1)
    wrapped_model.set_inputs(as_variable(decoder_input), as_variable(t), t)
    return x, t


def accuracy_fn(_y, _t) -> Variable:
    """Compute accuracy using wrapped_model's stored state."""
    acc = wrapped_model.compute_accuracy()
    return as_variable(np.array(acc))


# %%
# Create Trainer and run
trainer = Trainer(
    model=wrapped_model,
    optimizer=optimizer,
    loss_fn=lambda y, _t: y,
    metric_fn=accuracy_fn,
    train_loader=train_loader,
    test_loader=test_loader,
    max_epoch=max_epoch,
    preprocess_fn=preprocess_fn,
    max_grad=max_grad,
)

print("Starting training...")
trainer.run()


# %%
# Save model weights
print("Saving model weights...")
model.save_weights("date_format_attention.npz")
print("Model saved to 'date_format_attention.npz'")


# %%
# Plot training history
trainer.plot_history(
    history_types=["loss", "test_loss"],
    title="Attention Seq2Seq Training Loss (Date Format)",
    figsize=(12, 6),
)

trainer.plot_history(
    history_types=["metric", "test_metric"],
    ylabel="Accuracy",
    title="Attention Seq2Seq Accuracy (Date Format)",
    figsize=(12, 6),
)


# %%
# Test examples
print("=" * 60)
print("Test Examples")
print("=" * 60)

with dpl.no_grad():
    for i in range(min(10, len(test_set))):
        x, t = test_set[i]
        x_var = as_variable(x.reshape(1, -1))

        generated = model.generate(x_var, start_id, len(t))
        pred_str = datasets.DateFormat.decode(generated[0])
        target_str = datasets.DateFormat.decode(t)
        input_str = datasets.DateFormat.decode(x)

        status = "✓" if pred_str.strip() == target_str.strip() else "✗"
        print(
            f"{status} Input: '{input_str}' -> Pred: '{pred_str}' (Target: '{target_str}')"
        )


# %%
# Visualize attention weights
import matplotlib.pyplot as plt


def visualize_attention(model, x, generated, ax=None):
    """Visualize attention weights for a single example.

    Args:
        model: AttentionSeq2SeqWithLoss model (must have called generate() first)
        x: Input sequence (numpy array)
        generated: Generated sequence from model.generate()
        ax: Matplotlib axis (optional)
    """
    # Get attention weights from last generate() call
    attention_weights = model.attention_weights
    attention_matrix = np.stack([a.data_required[0] for a in attention_weights], axis=0)

    # Get input/output strings
    input_chars = list(datasets.DateFormat.decode(x))
    output_chars = [datasets.DateFormat.ID2CHAR[int(g)] for g in generated]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(attention_matrix, cmap="Blues", aspect="auto", vmin=0.0)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(input_chars)))
    ax.set_xticklabels(input_chars, fontsize=8)
    ax.set_yticks(range(len(output_chars)))
    ax.set_yticklabels(output_chars, fontsize=10)

    ax.set_xlabel("Input")
    ax.set_ylabel("Output")

    return im


# %%
# Plot attention for several examples
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

with dpl.no_grad():
    for idx, ax in enumerate(axes):
        x, t = test_set[idx]
        x_var = as_variable(x.reshape(1, -1))
        generated = model.generate(x_var, start_id, len(t))
        visualize_attention(model, x, generated[0], ax=ax)

        input_str = datasets.DateFormat.decode(x)
        pred_str = datasets.DateFormat.decode(generated[0])
        target_str = datasets.DateFormat.decode(t)
        status = "✓" if pred_str.strip() == target_str.strip() else "✗"
        ax.set_title(f"{status} '{input_str.strip()}' -> '{pred_str}'", fontsize=10)

plt.tight_layout()
plt.show()
