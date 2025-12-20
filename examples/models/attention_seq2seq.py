"""
Attention Seq2Seq model implementation.

This module provides Encoder-Decoder architecture with attention mechanism
for sequence-to-sequence tasks.
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
        self.embed = L.Embedding(vocab_size, embedding_dim)
        self.lstm = L.LSTM(hidden_size, in_size=embedding_dim)
        self.affine = L.Linear(vocab_size, in_size=hidden_size * 2)
        self._encoder_hs = None
        self._last_attention_weights: list[Variable] = []

    def step(self, token: Variable) -> Variable:
        """
        Process a single timestep.

        Args:
            token: Input token (batch_size,)

        Returns:
            scores: Output scores (batch_size, vocab_size)
        """
        if self._encoder_hs is None:
            raise ValueError("encoder_hs must be set before step")

        # Embed token
        embedded = self.embed(token)  # (batch_size, embedding_dim)

        # LSTM step
        h = self.lstm(embedded)  # (batch_size, hidden_size)

        # Attention
        context, a = F.attention(
            self._encoder_hs, h
        )  # (batch_size, hidden_size), (batch_size, seq_len)

        # Store attention weights
        self._last_attention_weights.append(a)

        # Concatenate and project
        concat = F.concat([h, context], axis=1)  # (batch_size, hidden_size * 2)
        scores = self.affine(concat)  # (batch_size, vocab_size)

        return scores

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Target sequence for teacher forcing (batch_size, seq_len)

        Returns:
            Output scores (batch_size, seq_len, vocab_size)
        """
        (x,) = xs
        batch_size, seq_len = x.shape

        all_scores = []
        self._last_attention_weights = []  # Clear before starting
        for t in range(seq_len):
            token = x[:, t]  # (batch_size,)
            scores = self.step(token)  # (batch_size, vocab_size)
            # Reshape for concat: (batch_size, vocab_size) -> (batch_size, 1, vocab_size)
            scores_3d = F.reshape(scores, (batch_size, 1, scores.shape[1]))
            all_scores.append(scores_3d)

        # Concat along time axis: (batch_size, seq_len, vocab_size)
        return F.concat(all_scores, axis=1)

    @property
    def attention_weights(self) -> list[Variable]:
        """Get attention weights from last forward pass."""
        return self._last_attention_weights

    def set_state(self, h, c, encoder_hs):
        """Set initial LSTM state and encoder hidden states."""
        self.lstm.h = h
        self.lstm.c = c
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
        self.decoder._last_attention_weights = []  # Clear before starting
        current_token = as_variable(np.full((batch_size,), start_id, dtype=np.int32))

        with dpl.no_grad():
            for _ in range(max_len):
                # Use decoder.step() - same as training
                scores = self.decoder.step(current_token)

                # Greedy decoding
                next_token = np.argmax(scores.data_required, axis=1)
                generated.append(next_token)

                # Prepare next input
                current_token = as_variable(next_token.astype(np.int32))

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
    def attention_weights(self) -> list[Variable]:
        """Get attention weights from last forward pass."""
        return self.seq2seq.decoder.attention_weights
