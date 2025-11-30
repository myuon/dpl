"""Tests for RNN layer."""

import numpy as np
import pytest

from dpl.core import as_variable
import dpl.layers as L
import dpl.functions as F


class TestRNN:
    def test_init(self):
        """Test RNN initialization."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        assert rnn.h is None

    def test_forward_first_step(self):
        """Test RNN forward pass on first time step."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        # Input: batch_size=2, in_size=5
        x = as_variable(np.random.randn(2, 5))

        h = rnn(x)

        # Output should have shape (batch_size, hidden_size)
        assert h.shape == (2, 10)
        assert rnn.h is not None
        assert rnn.h.shape == (2, 10)

    def test_forward_multiple_steps(self):
        """Test RNN forward pass over multiple time steps."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        # Sequence of 3 time steps
        x1 = as_variable(np.random.randn(2, 5))
        x2 = as_variable(np.random.randn(2, 5))
        x3 = as_variable(np.random.randn(2, 5))

        h1 = rnn(x1)
        h2 = rnn(x2)
        h3 = rnn(x3)

        # All outputs should have the same shape
        assert h1.shape == (2, 10)
        assert h2.shape == (2, 10)
        assert h3.shape == (2, 10)

        # Hidden states should be different at each step
        assert not np.allclose(h1.data, h2.data)
        assert not np.allclose(h2.data, h3.data)

    def test_reset_state(self):
        """Test state reset functionality."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        # Process some input
        x = as_variable(np.random.randn(2, 5))
        rnn(x)

        assert rnn.h is not None

        # Reset state
        rnn.reset_state()

        assert rnn.h is None

    def test_state_persistence(self):
        """Test that state persists across forward passes."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        x1 = as_variable(np.random.randn(2, 5))
        x2 = as_variable(np.random.randn(2, 5))

        h1 = rnn(x1)
        h1_saved = rnn.h.data.copy()

        h2 = rnn(x2)

        # State should have changed after second input
        assert not np.allclose(rnn.h.data, h1_saved)

    def test_backward(self):
        """Test RNN backward pass."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        x = as_variable(np.random.randn(2, 5))
        h = rnn(x)

        # Compute some loss
        loss = F.sum(h)

        # Backward should not raise errors
        loss.backward()

        # Gradients should be computed for parameters that were used
        params = list(rnn.params())
        assert len(params) > 0

        # At least some parameters should have gradients
        params_with_grad = [p for p in params if p.grad is not None]
        assert len(params_with_grad) > 0

    def test_sequence_processing(self):
        """Test processing a sequence and computing loss."""
        rnn = L.RNN(hidden_size=10, in_size=5)
        linear = L.Linear(1, in_size=10)

        sequence_length = 5
        batch_size = 2

        total_loss = as_variable(np.array(0.0))

        for t in range(sequence_length):
            x = as_variable(np.random.randn(batch_size, 5))
            h = rnn(x)
            y = linear(h)
            target = as_variable(np.random.randn(batch_size, 1))
            loss = F.mean_squared_error(y, target)
            total_loss += loss

        # Backward through entire sequence
        total_loss.backward()

        # All parameters should have gradients
        for param in rnn.params():
            assert param.grad is not None

        for param in linear.params():
            assert param.grad is not None

    def test_params(self):
        """Test parameter counting."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        params = list(rnn.params())

        # RNN has:
        # - x2h: weight + bias = 2 params
        # - h2h: weight (no bias) = 1 param
        # Total: 3 parameters
        assert len(params) == 3

    def test_cleargrads(self):
        """Test gradient clearing."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        x = as_variable(np.random.randn(2, 5))
        h = rnn(x)
        loss = F.sum(h)
        loss.backward()

        # At least some gradients should exist
        params_with_grad = [p for p in rnn.params() if p.grad is not None]
        assert len(params_with_grad) > 0

        # Clear gradients
        rnn.cleargrads()

        # All gradients should be None after clearing
        for param in rnn.params():
            assert param.grad is None

    def test_batch_sizes(self):
        """Test RNN with different batch sizes."""
        rnn = L.RNN(hidden_size=10, in_size=5)

        # Batch size 1
        x1 = as_variable(np.random.randn(1, 5))
        h1 = rnn(x1)
        assert h1.shape == (1, 10)

        rnn.reset_state()

        # Batch size 5
        x5 = as_variable(np.random.randn(5, 5))
        h5 = rnn(x5)
        assert h5.shape == (5, 10)

    def test_hidden_sizes(self):
        """Test RNN with different hidden sizes."""
        for hidden_size in [1, 10, 50, 100]:
            rnn = L.RNN(hidden_size=hidden_size, in_size=5)
            x = as_variable(np.random.randn(2, 5))
            h = rnn(x)
            assert h.shape == (2, hidden_size)
            assert rnn.h.shape == (2, hidden_size)

    def test_rnn_computation_matches_numpy(self):
        """Test that RNN computation matches numpy implementation.

        Verify that h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        computed directly with numpy matches the RNN layer forward pass.
        """
        hidden_size = 10
        in_size = 5
        batch_size = 2

        # Create RNN layer
        rnn = L.RNN(hidden_size=hidden_size, in_size=in_size)

        # Get the weights and bias from the RNN layer
        # x2h contains W_xh and b_h
        W_xh = rnn.x2h.W.data  # shape: (in_size, hidden_size)
        b_h = rnn.x2h.b.data   # shape: (hidden_size,)
        # h2h contains W_hh (no bias)
        W_hh = rnn.h2h.W.data  # shape: (hidden_size, hidden_size)

        # Test first step (no previous hidden state)
        x1 = np.random.randn(batch_size, in_size)

        # Compute with RNN layer
        h1_rnn = rnn(as_variable(x1))

        # Compute with numpy: h_t = tanh(x_t @ W_xh + b_h)
        h1_numpy = np.tanh(x1 @ W_xh + b_h)

        # Should be very close
        assert np.allclose(h1_rnn.data, h1_numpy, rtol=1e-5, atol=1e-7)

        # Test second step (with previous hidden state)
        x2 = np.random.randn(batch_size, in_size)

        # Compute with RNN layer
        h2_rnn = rnn(as_variable(x2))

        # Compute with numpy: h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
        h2_numpy = np.tanh(x2 @ W_xh + h1_numpy @ W_hh + b_h)

        # Should be very close
        assert np.allclose(h2_rnn.data, h2_numpy, rtol=1e-5, atol=1e-7)

    def test_rnn_multiple_steps_matches_numpy(self):
        """Test multiple RNN steps match numpy implementation."""
        hidden_size = 8
        in_size = 6
        batch_size = 3
        num_steps = 5

        # Create RNN layer
        rnn = L.RNN(hidden_size=hidden_size, in_size=in_size)

        # Get weights
        W_xh = rnn.x2h.W.data
        b_h = rnn.x2h.b.data
        W_hh = rnn.h2h.W.data

        # Generate random inputs
        xs = [np.random.randn(batch_size, in_size) for _ in range(num_steps)]

        # Compute with RNN layer
        rnn.reset_state()
        hs_rnn = []
        for x in xs:
            h = rnn(as_variable(x))
            hs_rnn.append(h.data)

        # Compute with numpy
        hs_numpy = []
        h_prev = None
        for x in xs:
            if h_prev is None:
                h = np.tanh(x @ W_xh + b_h)
            else:
                h = np.tanh(x @ W_xh + h_prev @ W_hh + b_h)
            hs_numpy.append(h)
            h_prev = h

        # All steps should match
        for h_rnn, h_numpy in zip(hs_rnn, hs_numpy):
            assert np.allclose(h_rnn, h_numpy, rtol=1e-5, atol=1e-7)
