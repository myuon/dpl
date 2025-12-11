# %%
"""
Seq2Seq model for addition expression learning.

This script trains a Seq2Seq model to learn addition:
    Input: "123+456" -> Output: "579"
"""
import numpy as np
import dpl
import dpl.functions as F
import dpl.layers as L
import dpl.optimizers as O
from dpl.core import as_variable, Variable
from dpl.dataloaders import DataLoader
from dpl.trainer import Trainer
from dpl.layers import Layer
import datasets


# %%
class Encoder(Layer):
    """
    Encoder for Seq2Seq model.

    Processes input sequence and outputs final hidden state.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embed = L.TimeEmbedding(vocab_size, embedding_dim)
        self.lstm = L.TimeLSTM(hidden_size, in_size=embedding_dim, stateful=False)

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Input sequence (batch_size, seq_len)

        Returns:
            Final hidden state (batch_size, hidden_size)
        """
        (x,) = xs
        # x: (batch_size, seq_len)
        embedded = self.embed(x)  # (batch_size, seq_len, embedding_dim)
        hs = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        # Return final hidden state
        return hs[:, -1, :]  # (batch_size, hidden_size)

    def get_state(self):
        """Get encoder's final LSTM state (h, c)."""
        return self.lstm.lstm.h, self.lstm.lstm.c


class Decoder(Layer):
    """
    Decoder for Seq2Seq model.

    Generates output sequence from encoder hidden state.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embed = L.TimeEmbedding(vocab_size, embedding_dim)
        self.lstm = L.TimeLSTM(hidden_size, in_size=embedding_dim, stateful=True)
        self.affine = L.TimeAffine(vocab_size, in_size=hidden_size)

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Target sequence for teacher forcing (batch_size, seq_len)

        Returns:
            Output scores (batch_size, seq_len, vocab_size)
        """
        (x,) = xs
        # x: (batch_size, seq_len)
        embedded = self.embed(x)  # (batch_size, seq_len, embedding_dim)
        hs = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        scores = self.affine(hs)  # (batch_size, seq_len, vocab_size)
        return scores

    def set_state(self, h, c):
        """Set initial LSTM state from encoder."""
        self.lstm.lstm.h = h
        self.lstm.lstm.c = c

    def reset_state(self):
        """Reset LSTM state."""
        self.lstm.reset_state()


class Seq2Seq(Layer):
    """
    Seq2Seq model combining encoder and decoder.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size)
        self.vocab_size = vocab_size

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: (input_seq, target_seq)
                - input_seq: (batch_size, input_len)
                - target_seq: (batch_size, output_len) - for teacher forcing

        Returns:
            Output scores (batch_size, output_len, vocab_size)
        """
        input_seq, target_seq = xs

        # Encode
        _ = self.encoder(input_seq)
        h, c = self.encoder.get_state()

        # Set decoder initial state
        self.decoder.set_state(h, c)

        # Decode with teacher forcing
        scores = self.decoder(target_seq)

        return scores

    def generate(self, input_seq: Variable, start_id: int, max_len: int) -> np.ndarray:
        """
        Generate output sequence (inference mode).

        Args:
            input_seq: Input sequence (batch_size, input_len)
            start_id: Start token ID
            max_len: Maximum output length

        Returns:
            Generated sequence (batch_size, max_len)
        """
        batch_size = input_seq.shape[0]

        # Encode
        _ = self.encoder(input_seq)
        h, c = self.encoder.get_state()
        self.decoder.set_state(h, c)

        # Generate one token at a time
        generated = []
        current_input = as_variable(
            np.full((batch_size, 1), start_id, dtype=np.int32)
        )

        with dpl.no_grad():
            for _ in range(max_len):
                # Get next token scores
                embedded = self.decoder.embed(current_input)
                h = self.decoder.lstm.lstm(embedded[:, 0, :])
                scores = self.decoder.affine.linear(h)  # (batch_size, vocab_size)

                # Greedy decoding
                next_token = np.argmax(scores.data_required, axis=1)
                generated.append(next_token)

                # Prepare next input
                current_input = as_variable(
                    next_token.reshape(batch_size, 1).astype(np.int32)
                )

        return np.stack(generated, axis=1)  # (batch_size, max_len)


class Seq2SeqWithLoss(Layer):
    """
    Seq2Seq model with built-in loss computation.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.seq2seq = Seq2Seq(vocab_size, embedding_dim, hidden_size)
        self.loss_layer = L.TimeSoftmaxWithLoss()

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: (input_seq, decoder_input, target)
                - input_seq: (batch_size, input_len)
                - decoder_input: (batch_size, output_len) - shifted target for teacher forcing
                - target: (batch_size, output_len) - ground truth

        Returns:
            Loss value
        """
        input_seq, decoder_input, target = xs
        scores = self.seq2seq(input_seq, decoder_input)
        loss = self.loss_layer(scores, target)
        return loss

    def generate(self, input_seq: Variable, start_id: int, max_len: int) -> np.ndarray:
        """Generate output sequence."""
        return self.seq2seq.generate(input_seq, start_id, max_len)


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
    print(f"  Input: '{datasets.AddExpr.decode(x)}' -> Target: '{datasets.AddExpr.decode(t)}'")


# %%
# Hyperparameters
vocab_size = datasets.AddExpr.VOCAB_SIZE
embedding_dim = 16
hidden_size = 128
max_epoch = 30
batch_size = 128
lr = 0.001

print("\n" + "=" * 60)
print("Training Configuration")
print("=" * 60)
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Hidden size: {hidden_size}")
print(f"Max epoch: {max_epoch}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print("=" * 60 + "\n")


# %%
# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# %%
# Create model
model = Seq2SeqWithLoss(vocab_size, embedding_dim, hidden_size)
optimizer = O.Adam(lr=lr).setup(model)

print("Model created successfully!")
print(f"Model parameters: {sum(p.data.size for p in model.params() if p.data is not None)}")


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
        print(f"{status} Input: '{input_str}' -> Pred: '{pred_str}' (Target: '{target_str}')")


# %%
# Interactive testing
def test_addition(a: int, b: int):
    """Test the model with custom numbers."""
    input_str = f"{a}+{b}".rjust(train_set.input_len)
    input_ids = datasets.AddExpr.encode(input_str)

    with dpl.no_grad():
        x_var = as_variable(input_ids.reshape(1, -1))
        generated = model.generate(x_var, start_id, train_set.output_len)
        pred_str = datasets.AddExpr.decode(generated[0])

    expected = str(a + b).rjust(train_set.output_len)
    status = "✓" if pred_str == expected else "✗"
    print(f"{status} {a} + {b} = {pred_str.strip()} (expected: {expected.strip()})")


print("\n" + "=" * 60)
print("Custom Test Examples")
print("=" * 60)
test_addition(123, 456)
test_addition(999, 1)
test_addition(0, 0)
test_addition(500, 500)
test_addition(111, 222)
