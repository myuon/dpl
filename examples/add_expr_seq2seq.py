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
train_set = datasets.AddExpr(num_samples=50000, max_digits=3, train=True, seed=42)
test_set = datasets.AddExpr(num_samples=1000, max_digits=3, train=False, seed=42)

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
# Custom training loop for Seq2Seq
# (Trainer expects model(x) -> y, but Seq2Seq needs (input, decoder_input, target))
def train_epoch(model, optimizer, train_loader, pad_id, start_id):
    """Train for one epoch."""
    total_loss = 0.0
    batch_count = 0

    for x_batch, t_batch in train_loader:
        # x_batch: (batch_size, input_len) - encoder input
        # t_batch: (batch_size, output_len) - target output

        # Create decoder input: prepend start token, remove last token
        batch_size = x_batch.shape[0]
        start_tokens = np.full((batch_size, 1), start_id, dtype=np.int32)
        decoder_input = np.concatenate([start_tokens, t_batch[:, :-1]], axis=1)

        # Convert to Variables
        x = as_variable(x_batch)
        dec_in = as_variable(decoder_input)
        t = as_variable(t_batch)

        # Forward
        loss = model(x, dec_in, t)

        # Backward
        model.cleargrads()
        loss.backward()
        optimizer.update()

        total_loss += loss.data_required.astype(float).item()
        batch_count += 1

    return total_loss / batch_count


def evaluate(model, test_loader, pad_id, start_id):
    """Evaluate model on test set."""
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    with dpl.no_grad():
        for x_batch, t_batch in test_loader:
            batch_size = x_batch.shape[0]
            start_tokens = np.full((batch_size, 1), start_id, dtype=np.int32)
            decoder_input = np.concatenate([start_tokens, t_batch[:, :-1]], axis=1)

            x = as_variable(x_batch)
            dec_in = as_variable(decoder_input)
            t = as_variable(t_batch)

            # Loss
            loss = model(x, dec_in, t)
            total_loss += loss.data_required.astype(float).item()
            batch_count += 1

            # Accuracy (sequence-level)
            generated = model.generate(x, start_id, t_batch.shape[1])
            correct += np.sum(np.all(generated == t_batch, axis=1))
            total += batch_size

    return total_loss / batch_count, correct / total


# %%
# Training
print("Starting training...")
print("-" * 60)

pad_id = datasets.AddExpr.PAD_ID
start_id = datasets.AddExpr.START_ID

train_loss_history = []
test_loss_history = []
test_acc_history = []

for epoch in range(max_epoch):
    train_loss = train_epoch(model, optimizer, train_loader, pad_id, start_id)
    test_loss, test_acc = evaluate(model, test_loader, pad_id, start_id)

    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    print(
        f"Epoch {epoch + 1}/{max_epoch}, "
        f"train_loss: {train_loss:.4f}, "
        f"test_loss: {test_loss:.4f}, "
        f"test_acc: {test_acc:.4f}"
    )


# %%
# Plot training history
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(train_loss_history, label="Train", marker="o")
axes[0].plot(test_loss_history, label="Test", marker="s")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Seq2Seq Training Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(test_acc_history, label="Test Accuracy", marker="o", color="green")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Seq2Seq Test Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


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
