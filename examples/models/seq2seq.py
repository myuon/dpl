"""
Seq2Seq model implementation.

This module provides Encoder-Decoder architecture for sequence-to-sequence tasks.
"""

import numpy as np
import dpl
import dpl.functions as F
import dpl.layers as L
from dpl.core import as_variable, Variable
from dpl.layers import Layer


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


class PeekyDecoder(Layer):
    """
    Peeky Decoder for Seq2Seq model.

    Concatenates encoder's hidden state to each time step's input for both
    LSTM and Affine layers. This allows the decoder to "peek" at the encoder's
    final state at every step.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = L.TimeEmbedding(vocab_size, embedding_dim)
        # LSTM input: embedding + encoder_h
        self.lstm = L.TimeLSTM(
            hidden_size, in_size=embedding_dim + hidden_size, stateful=True
        )
        # Affine input: lstm_output + encoder_h
        self.affine = L.TimeAffine(vocab_size, in_size=hidden_size + hidden_size)
        self._encoder_h = None

    def forward(self, *xs: Variable) -> Variable:
        """
        Args:
            xs: Target sequence for teacher forcing (batch_size, seq_len)

        Returns:
            Output scores (batch_size, seq_len, vocab_size)
        """
        (x,) = xs
        batch_size, seq_len = x.shape[0], x.shape[1]

        if self._encoder_h is None:
            raise ValueError("encoder_h must be set before forward")

        # x: (batch_size, seq_len)
        embedded = self.embed(x)  # (batch_size, seq_len, embedding_dim)

        # Repeat encoder_h for each time step
        # encoder_h: (batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
        encoder_h_repeated = F.broadcast_to(
            F.reshape(self._encoder_h, (batch_size, 1, self.hidden_size)),
            (batch_size, seq_len, self.hidden_size),
        )

        # Concat embedding with encoder_h for LSTM input
        lstm_input = F.concat([embedded, encoder_h_repeated], axis=2)
        hs = self.lstm(lstm_input)  # (batch_size, seq_len, hidden_size)

        # Concat LSTM output with encoder_h for Affine input
        affine_input = F.concat([hs, encoder_h_repeated], axis=2)
        scores = self.affine(affine_input)  # (batch_size, seq_len, vocab_size)

        return scores

    def set_state(self, h, c):
        """Set initial LSTM state and encoder hidden state."""
        self.lstm.lstm.h = h
        self.lstm.lstm.c = c
        self._encoder_h = h

    def reset_state(self):
        """Reset LSTM state."""
        self.lstm.reset_state()
        self._encoder_h = None


class Seq2Seq(Layer):
    """
    Seq2Seq model combining encoder and decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        peeky: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size)
        self.peeky = peeky
        self.hidden_size = hidden_size
        if peeky:
            self.decoder = PeekyDecoder(vocab_size, embedding_dim, hidden_size)
        else:
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
        current_input = as_variable(np.full((batch_size, 1), start_id, dtype=np.int32))

        with dpl.no_grad():
            for _ in range(max_len):
                # Get next token scores
                embedded = self.decoder.embed(current_input)

                if self.peeky:
                    # Peeky: concat encoder_h to embedding
                    encoder_h = self.decoder._encoder_h
                    lstm_in = F.concat([embedded[:, 0, :], encoder_h], axis=1)
                    h = self.decoder.lstm.lstm(lstm_in)
                    # Peeky: concat encoder_h to LSTM output
                    affine_in = F.concat([h, encoder_h], axis=1)
                    scores = self.decoder.affine.linear(affine_in)
                else:
                    h = self.decoder.lstm.lstm(embedded[:, 0, :])
                    scores = self.decoder.affine.linear(h)

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

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        peeky: bool = False,
    ):
        super().__init__()
        self.seq2seq = Seq2Seq(vocab_size, embedding_dim, hidden_size, peeky=peeky)
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
