"""Tests for TimeRNN layer."""

import numpy as np
import pytest

from dpl.core import as_variable
import dpl.layers as L
from dpl.layers.time_layers import TimeRNN
import dpl.functions as F


class TestTimeRNN:
    def test_init(self):
        """Test TimeRNN initialization."""
        time_rnn = TimeRNN(hidden_size=10, in_size=5)

        assert time_rnn.rnn.h is None
        assert time_rnn.stateful is False

    def test_init_stateful(self):
        """Test TimeRNN initialization with stateful=True."""
        time_rnn = TimeRNN(hidden_size=10, in_size=5, stateful=True)

        assert time_rnn.stateful is True

    def test_forward_shape(self):
        """Test TimeRNN forward pass output shape."""
        batch_size = 2
        seq_len = 5
        in_size = 7
        hidden_size = 10

        time_rnn = TimeRNN(hidden_size=hidden_size, in_size=in_size)

        # Input: (batch_size, seq_len, in_size)
        x = as_variable(np.random.randn(batch_size, seq_len, in_size))

        # Forward pass
        h = time_rnn(x)

        # Output should have shape (batch_size, seq_len, hidden_size)
        assert h.shape == (batch_size, seq_len, hidden_size)

    def test_reset_state(self):
        """Test state reset functionality."""
        time_rnn = TimeRNN(hidden_size=10, in_size=5, stateful=True)

        # Process some input
        x = as_variable(np.random.randn(2, 3, 5))
        time_rnn(x)

        assert time_rnn.rnn.h is not None

        # Reset state
        time_rnn.reset_state()

        assert time_rnn.rnn.h is None

    def test_stateful_false_resets_state(self):
        """Test that stateful=False resets state at each forward pass."""
        time_rnn = TimeRNN(hidden_size=10, in_size=5, stateful=False)

        # First forward pass
        x1 = as_variable(np.random.randn(2, 3, 5))
        h1 = time_rnn(x1)

        # Second forward pass - should reset state automatically
        x2 = as_variable(np.random.randn(2, 3, 5))
        h2 = time_rnn(x2)

        # With same input but reset state, first time step should be the same
        time_rnn.reset_state()
        x_same = x2
        h_same = time_rnn(x_same)

        # First time step should be identical since state was reset both times
        assert np.allclose(h2.data[:, 0, :], h_same.data[:, 0, :])

    def test_stateful_true_maintains_state(self):
        """Test that stateful=True maintains state across forward passes."""
        time_rnn = TimeRNN(hidden_size=10, in_size=5, stateful=True)

        # First forward pass
        x1 = as_variable(np.random.randn(2, 3, 5))
        h1 = time_rnn(x1)
        h_saved = time_rnn.rnn.h.data.copy()

        # Second forward pass - should NOT reset state
        x2 = as_variable(np.random.randn(2, 3, 5))
        h2 = time_rnn(x2)

        # State should be different from saved state
        assert not np.allclose(time_rnn.rnn.h.data, h_saved)

    def test_backward(self):
        """Test TimeRNN backward pass."""
        time_rnn = TimeRNN(hidden_size=10, in_size=5)

        x = as_variable(np.random.randn(2, 3, 5))
        h = time_rnn(x)

        # Compute some loss
        loss = F.sum(h)

        # Backward should not raise errors
        loss.backward()

        # Gradients should be computed for parameters
        params = list(time_rnn.params())
        assert len(params) > 0

        params_with_grad = [p for p in params if p.grad is not None]
        assert len(params_with_grad) > 0

    def test_params(self):
        """Test parameter counting."""
        time_rnn = TimeRNN(hidden_size=10, in_size=5)

        params = list(time_rnn.params())

        # TimeRNN wraps RNN which has:
        # - x2h: weight + bias = 2 params
        # - h2h: weight (no bias) = 1 param
        # Total: 3 parameters
        assert len(params) == 3

    def test_time_rnn_matches_numpy_loop(self):
        """Test that TimeRNN matches numpy implementation with loop.

        Verify that processing a sequence with TimeRNN produces the same result
        as manually looping through time steps with numpy computation:
        h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
        """
        batch_size = 2
        seq_len = 5
        in_size = 6
        hidden_size = 8

        # Create TimeRNN layer
        time_rnn = TimeRNN(hidden_size=hidden_size, in_size=in_size)

        # Get weights from the underlying RNN
        W_xh = time_rnn.rnn.x2h.W.data  # shape: (in_size, hidden_size)
        b_h = time_rnn.rnn.x2h.b.data   # shape: (hidden_size,)
        W_hh = time_rnn.rnn.h2h.W.data  # shape: (hidden_size, hidden_size)

        # Generate random input sequence
        x = np.random.randn(batch_size, seq_len, in_size)

        # Compute with TimeRNN layer
        h_time_rnn = time_rnn(as_variable(x))

        # Compute with numpy loop
        hs_numpy = []
        h_prev = None
        for t in range(seq_len):
            x_t = x[:, t, :]  # shape: (batch_size, in_size)
            if h_prev is None:
                # First time step: h_t = tanh(x_t @ W_xh + b_h)
                h_t = np.tanh(x_t @ W_xh + b_h)
            else:
                # Subsequent steps: h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
                h_t = np.tanh(x_t @ W_xh + h_prev @ W_hh + b_h)
            hs_numpy.append(h_t)
            h_prev = h_t

        # Stack numpy results: (seq_len, batch_size, hidden_size)
        hs_numpy_stacked = np.stack(hs_numpy, axis=0)
        # Transpose to (batch_size, seq_len, hidden_size)
        hs_numpy_stacked = np.transpose(hs_numpy_stacked, (1, 0, 2))

        # Should be very close
        assert np.allclose(h_time_rnn.data, hs_numpy_stacked, rtol=1e-5, atol=1e-7)

    def test_time_rnn_matches_manual_rnn_loop(self):
        """Test that TimeRNN matches manually looping through RNN layer.

        Verify that TimeRNN produces the same result as manually calling
        the RNN layer for each time step.
        """
        batch_size = 3
        seq_len = 4
        in_size = 5
        hidden_size = 7

        # Create two RNN instances with same weights
        time_rnn = TimeRNN(hidden_size=hidden_size, in_size=in_size)
        manual_rnn = L.RNN(hidden_size=hidden_size, in_size=in_size)

        # Copy weights from TimeRNN to manual RNN
        manual_rnn.x2h.W.data[:] = time_rnn.rnn.x2h.W.data
        manual_rnn.x2h.b.data[:] = time_rnn.rnn.x2h.b.data
        manual_rnn.h2h.W.data[:] = time_rnn.rnn.h2h.W.data

        # Generate random input sequence
        x = np.random.randn(batch_size, seq_len, in_size)

        # Compute with TimeRNN layer
        h_time_rnn = time_rnn(as_variable(x))

        # Compute with manual RNN loop
        manual_rnn.reset_state()
        hs_manual = []
        for t in range(seq_len):
            x_t = as_variable(x[:, t, :])  # (batch_size, in_size)
            h_t = manual_rnn(x_t)
            hs_manual.append(h_t.data)

        # Stack manual results: (seq_len, batch_size, hidden_size)
        hs_manual_stacked = np.stack(hs_manual, axis=0)
        # Transpose to (batch_size, seq_len, hidden_size)
        hs_manual_stacked = np.transpose(hs_manual_stacked, (1, 0, 2))

        # Should be exactly the same
        assert np.allclose(h_time_rnn.data, hs_manual_stacked, rtol=1e-5, atol=1e-7)

    def test_time_rnn_different_sequence_lengths(self):
        """Test TimeRNN with different sequence lengths."""
        batch_size = 2
        hidden_size = 10
        in_size = 5

        time_rnn = TimeRNN(hidden_size=hidden_size, in_size=in_size)

        for seq_len in [1, 5, 10, 20]:
            x = as_variable(np.random.randn(batch_size, seq_len, in_size))
            h = time_rnn(x)
            assert h.shape == (batch_size, seq_len, hidden_size)

    def test_time_rnn_batch_sizes(self):
        """Test TimeRNN with different batch sizes."""
        seq_len = 5
        hidden_size = 10
        in_size = 5

        time_rnn = TimeRNN(hidden_size=hidden_size, in_size=in_size)

        for batch_size in [1, 2, 8, 16]:
            time_rnn.reset_state()
            x = as_variable(np.random.randn(batch_size, seq_len, in_size))
            h = time_rnn(x)
            assert h.shape == (batch_size, seq_len, hidden_size)

    def test_time_rnn_gradient_flow(self):
        """Test that gradients flow through time properly."""
        batch_size = 2
        seq_len = 5
        in_size = 5
        hidden_size = 10

        time_rnn = TimeRNN(hidden_size=hidden_size, in_size=in_size)

        x = as_variable(np.random.randn(batch_size, seq_len, in_size))
        h = time_rnn(x)

        # Loss depends on all time steps
        loss = F.sum(h)
        loss.backward()

        # All parameters should have gradients
        for param in time_rnn.params():
            assert param.grad is not None
            # Gradient should be non-zero
            assert not np.allclose(param.grad.data, 0.0)

    def test_cleargrads(self):
        """Test gradient clearing."""
        time_rnn = TimeRNN(hidden_size=10, in_size=5)

        x = as_variable(np.random.randn(2, 3, 5))
        h = time_rnn(x)
        loss = F.sum(h)
        loss.backward()

        # Clear gradients
        time_rnn.cleargrads()

        # All gradients should be None after clearing
        for param in time_rnn.params():
            assert param.grad is None
