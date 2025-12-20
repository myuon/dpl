# %%
"""
Seq2Seq model for addition expression learning.

This script trains a Seq2Seq model to learn addition:
    Input: "123+456" -> Output: "579"
"""
import numpy as np
import dpl
import dpl.optimizers as O
from dpl.core import as_variable, Variable
from dpl.dataloaders import DataLoader
from dpl.trainer import Trainer
from dpl.layers import Layer
import datasets
from models import Seq2SeqWithLoss


# %%
# Load dataset
print("Loading AddExpr dataset...")
train_set = datasets.AddExpr(
    num_samples=50000, max_digits=3, train=True, seed=42, reverse_input=True
)
test_set = datasets.AddExpr(
    num_samples=1000, max_digits=3, train=False, seed=42, reverse_input=True
)

print(f"Training samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")
print(f"Vocabulary size: {datasets.AddExpr.VOCAB_SIZE}")
print(f"Input length: {train_set.input_len}")
print(f"Output length: {train_set.output_len}")

# Show examples
print("\nExamples:")
for i in range(3):
    x, t = train_set[i]
    print(
        f"  Input: '{datasets.AddExpr.decode(x)}' -> Target: '{datasets.AddExpr.decode(t)}'"
    )


# %%
# Hyperparameters
vocab_size = datasets.AddExpr.VOCAB_SIZE
embedding_dim = 16
hidden_size = 128
max_epoch = 30
batch_size = 128
lr = 0.001
peeky = True

print("\n" + "=" * 60)
print("Training Configuration")
print("=" * 60)
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Hidden size: {hidden_size}")
print(f"Max epoch: {max_epoch}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Peeky decoder: {peeky}")
print("=" * 60 + "\n")


# %%
# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# %%
# Create model
model = Seq2SeqWithLoss(vocab_size, embedding_dim, hidden_size, peeky=peeky)
optimizer = O.Adam(lr=lr).setup(model)

print("Model created successfully!")
print(
    f"Model parameters: {sum(p.data.size for p in model.params() if p.data is not None)}"
)


# %%
# Wrapper to adapt Seq2SeqWithLoss for Trainer
class Seq2SeqWrapper(Layer):
    """
    Wrapper to adapt Seq2SeqWithLoss for Trainer.

    Trainer expects model(x) -> y, but Seq2Seq needs (input, decoder_input, target).
    This wrapper stores decoder_input and target via preprocess_fn.
    """

    def __init__(self, model: Seq2SeqWithLoss, start_id: int, output_len: int):
        super().__init__()
        self.model = model
        self.start_id = start_id
        self.output_len = output_len
        self._input = None
        self._decoder_input = None
        self._target = None
        self._target_np = None  # numpy array for accuracy calculation

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
        """Compute sequence-level accuracy using the stored input and target."""
        if self._input is None or self._target_np is None:
            return 0.0
        generated = self.model.generate(self._input, self.start_id, self.output_len)
        correct = np.sum(np.all(generated == self._target_np, axis=1))
        return correct / len(self._target_np)


# Wrap the model
start_id = datasets.AddExpr.START_ID
output_len = train_set.output_len
wrapped_model = Seq2SeqWrapper(model, start_id, output_len)
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
# Create Trainer
trainer = Trainer(
    model=wrapped_model,
    optimizer=optimizer,
    loss_fn=lambda y, _t: y,  # y is already the loss from Seq2SeqWithLoss
    metric_fn=accuracy_fn,
    train_loader=train_loader,
    test_loader=test_loader,
    max_epoch=max_epoch,
    preprocess_fn=preprocess_fn,
)

print("\nStarting training...")
trainer.run()


# %%
# Save model weights
print("\nSaving model weights...")
model.save_weights("add_expr_seq2seq.npz")
print("Model saved to 'add_expr_seq2seq.npz'")


# %%
# Plot training history
trainer.plot_history(
    history_types=["loss", "test_loss"],
    title="Seq2Seq Training Loss (Addition)",
    figsize=(12, 6),
)

trainer.plot_history(
    history_types=["metric", "test_metric"],
    ylabel="Accuracy",
    title="Seq2Seq Accuracy (Addition)",
    figsize=(12, 6),
)


# %%
# Test examples
print("\n" + "=" * 60)
print("Test Examples")
print("=" * 60)

# Get some test examples
with dpl.no_grad():
    for i in range(10):
        x, t = test_set[i]
        x_var = as_variable(x.reshape(1, -1))

        generated = model.generate(x_var, start_id, len(t))
        pred_str = datasets.AddExpr.decode(generated[0])
        target_str = datasets.AddExpr.decode(t)
        input_str = datasets.AddExpr.decode(x)

        status = "✓" if pred_str.strip() == target_str.strip() else "✗"
        print(
            f"{status} Input: '{input_str}' -> Pred: '{pred_str}' (Target: '{target_str}')"
        )


# %%
# Interactive testing
def test_addition(a: int, b: int):
    """Test the model with custom numbers."""
    input_ids = train_set.make_input(a, b)
    expected_ids = train_set.make_output(a + b)

    with dpl.no_grad():
        x_var = as_variable(input_ids.reshape(1, -1))
        generated = model.generate(x_var, start_id, train_set.output_len)
        pred_str = datasets.AddExpr.decode(generated[0])

    expected_str = datasets.AddExpr.decode(expected_ids)
    status = "✓" if pred_str == expected_str else "✗"
    print(f"{status} {a} + {b} = {pred_str.strip()} (expected: {expected_str.strip()})")


print("\n" + "=" * 60)
print("Custom Test Examples")
print("=" * 60)
test_addition(123, 456)
test_addition(999, 1)
test_addition(0, 0)
test_addition(500, 500)
test_addition(111, 222)
