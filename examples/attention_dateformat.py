"""
Attention Seq2Seq model implementation.
"""

import numpy as np
import dpl
import dpl.functions as F
import dpl.layers as L
from dpl.core import Variable, as_variable
from dpl.layers import Layer


class AttentionEncoder(Layer):
    """
    Encoder for Attention Seq2Seq model.

    Returns all hidden states for attention mechanism.
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
            All hidden states (batch_size, seq_len, hidden_size)
        """
        (x,) = xs
        embedded = self.embed(x)
        hs = self.lstm(embedded)
        return hs

    def get_state(self):
        """Get encoder's final LSTM state (h, c)."""
        return self.lstm.lstm.h, self.lstm.lstm.c


class AttentionDecoder(Layer):
    """
    Decoder with attention mechanism.

    At each time step, computes attention over encoder hidden states
    and concatenates context vector with LSTM output.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = L.TimeEmbedding(vocab_size, embedding_dim)
        self.lstm = L.TimeLSTM(hidden_size, in_size=embedding_dim, stateful=True)
        self.attention = L.TimeAttention()
        # Affine input: lstm_output + context_vector
        self.affine = L.TimeAffine(vocab_size, in_size=hidden_size * 2)
        self._encoder_hs = None

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Target sequence for teacher forcing (batch_size, seq_len)

        Returns:
            Output scores (batch_size, seq_len, vocab_size)
        """
        (x,) = xs

        if self._encoder_hs is None:
            raise ValueError("encoder_hs must be set before forward")

        embedded = self.embed(x)
        hs_dec = self.lstm(embedded)

        # Compute attention context vectors
        cs = self.attention(self._encoder_hs, hs_dec)

        # Concatenate LSTM output with context vectors
        concat = F.concat([hs_dec, cs], axis=2)
        scores = self.affine(concat)

        return scores

    def set_state(self, h, c, encoder_hs):
        """Set initial LSTM state and encoder hidden states."""
        self.lstm.lstm.h = h
        self.lstm.lstm.c = c
        self._encoder_hs = encoder_hs

    def reset_state(self):
        """Reset LSTM state."""
        self.lstm.reset_state()
        self._encoder_hs = None


class AttentionSeq2Seq(Layer):
    """
    Seq2Seq model with attention mechanism.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.encoder = AttentionEncoder(vocab_size, embedding_dim, hidden_size)
        self.decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

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

        # Encode - get all hidden states
        hs_enc = self.encoder(input_seq)
        h, c = self.encoder.get_state()

        # Set decoder initial state with encoder hidden states
        self.decoder.set_state(h, c, hs_enc)

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
        hs_enc = self.encoder(input_seq)
        h, c = self.encoder.get_state()
        self.decoder.set_state(h, c, hs_enc)

        # Generate one token at a time
        generated = []
        current_input = as_variable(np.full((batch_size, 1), start_id, dtype=np.int32))

        with dpl.no_grad():
            for _ in range(max_len):
                # Embed current input
                embedded = self.decoder.embed(current_input)

                # LSTM step
                h = self.decoder.lstm.lstm(embedded[:, 0, :])

                # Attention: compute context vector
                c, _a = F.attention(hs_enc, h)

                # Concatenate LSTM output with context
                concat = F.concat([h, c], axis=1)

                # Get scores
                scores = self.decoder.affine.linear(concat)

                # Greedy decoding
                next_token = np.argmax(scores.data_required, axis=1)
                generated.append(next_token)

                # Prepare next input
                current_input = as_variable(
                    next_token.reshape(batch_size, 1).astype(np.int32)
                )

        return np.stack(generated, axis=1)


class AttentionSeq2SeqWithLoss(Layer):
    """
    Attention Seq2Seq model with built-in loss computation.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.seq2seq = AttentionSeq2Seq(vocab_size, embedding_dim, hidden_size)
        self.loss_layer = L.TimeSoftmaxWithLoss()

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: (input_seq, decoder_input, target)
                - input_seq: (batch_size, input_len)
                - decoder_input: (batch_size, output_len) - shifted target
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

    @property
    def attention_weights(self):
        """Get attention weights from last forward pass."""
        return self.seq2seq.decoder.attention.attention_weights


# %%
import dpl.optimizers as O
from dpl.dataloaders import DataLoader
from dpl.trainer import Trainer
import datasets


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
for i in range(7):
    x, t = train_set[i]
    print(
        f"  Input: '{datasets.DateFormat.decode(x)}' -> Target: '{datasets.DateFormat.decode(t)}'"
    )


# %%
# Hyperparameters
vocab_size = datasets.DateFormat.VOCAB_SIZE
embedding_dim = 16
hidden_size = 128
max_epoch = 20
batch_size = 128
lr = 0.001

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
    for i in range(14):
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


def visualize_attention(model, x, t, ax=None):
    """Visualize attention weights for a single example."""
    x_var = as_variable(x.reshape(1, -1))

    # Get encoder hidden states
    hs_enc = model.seq2seq.encoder(x_var)
    h, c = model.seq2seq.encoder.get_state()
    model.seq2seq.decoder.set_state(h, c, hs_enc)

    # Generate and collect attention weights
    batch_size = 1
    generated = []
    attention_weights = []
    current_input = as_variable(np.full((batch_size, 1), start_id, dtype=np.int32))

    with dpl.no_grad():
        for _ in range(len(t)):
            embedded = model.seq2seq.decoder.embed(current_input)
            h = model.seq2seq.decoder.lstm.lstm(embedded[:, 0, :])
            context, a = F.attention(hs_enc, h)
            attention_weights.append(a.data_required[0])

            concat = F.concat([h, context], axis=1)
            scores = model.seq2seq.decoder.affine.linear(concat)
            next_token = np.argmax(scores.data_required, axis=1)
            generated.append(next_token[0])

            current_input = as_variable(
                next_token.reshape(batch_size, 1).astype(np.int32)
            )

    # Stack attention weights: (output_len, input_len)
    attention_matrix = np.stack(attention_weights, axis=0)

    # Get input/output strings
    input_chars = list(datasets.DateFormat.decode(x))
    output_chars = [datasets.DateFormat.ID2CHAR[int(g)] for g in generated]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(attention_matrix, cmap="Blues", aspect="auto")

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
        visualize_attention(model, x, t, ax=ax)

        input_str = datasets.DateFormat.decode(x)
        target_str = datasets.DateFormat.decode(t)
        ax.set_title(f"'{input_str.strip()}' -> '{target_str}'", fontsize=10)

plt.tight_layout()
plt.show()
print("Saved attention visualization to 'attention_visualization.png'")
