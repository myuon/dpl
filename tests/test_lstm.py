"""Tests for LSTM layer."""

import numpy as np
import pytest

from dpl.core import as_variable
import dpl.layers as L
import dpl.functions as F


class TestLSTM:
    def test_init(self):
        """Test LSTM initialization."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        assert lstm.hidden_size == 10
        assert lstm.h is None
        assert lstm.c is None

    def test_forward_first_step(self):
        """Test LSTM forward pass on first time step."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        # Input: batch_size=2, in_size=5
        x = as_variable(np.random.randn(2, 5))

        h = lstm(x)

        # Output should have shape (batch_size, hidden_size)
        assert h.shape == (2, 10)
        assert lstm.h is not None
        assert lstm.c is not None
        assert lstm.h.shape == (2, 10)
        assert lstm.c.shape == (2, 10)

    def test_forward_multiple_steps(self):
        """Test LSTM forward pass over multiple time steps."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        # Sequence of 3 time steps
        x1 = as_variable(np.random.randn(2, 5))
        x2 = as_variable(np.random.randn(2, 5))
        x3 = as_variable(np.random.randn(2, 5))

        h1 = lstm(x1)
        h2 = lstm(x2)
        h3 = lstm(x3)

        # All outputs should have the same shape
        assert h1.shape == (2, 10)
        assert h2.shape == (2, 10)
        assert h3.shape == (2, 10)

        # Hidden states should be different at each step
        assert not np.allclose(h1.data, h2.data)
        assert not np.allclose(h2.data, h3.data)

    def test_reset_state(self):
        """Test state reset functionality."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        # Process some input
        x = as_variable(np.random.randn(2, 5))
        lstm(x)

        assert lstm.h is not None
        assert lstm.c is not None

        # Reset state
        lstm.reset_state()

        assert lstm.h is None
        assert lstm.c is None

    def test_state_persistence(self):
        """Test that state persists across forward passes."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        x1 = as_variable(np.random.randn(2, 5))
        x2 = as_variable(np.random.randn(2, 5))

        h1 = lstm(x1)
        h1_saved = lstm.h.data.copy()
        c1_saved = lstm.c.data.copy()

        h2 = lstm(x2)

        # State should have changed after second input
        assert not np.allclose(lstm.h.data, h1_saved)
        assert not np.allclose(lstm.c.data, c1_saved)

    def test_backward(self):
        """Test LSTM backward pass."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        x = as_variable(np.random.randn(2, 5))
        h = lstm(x)

        # Compute some loss
        loss = F.sum(h)

        # Backward should not raise errors
        loss.backward()

        # Gradients should be computed for parameters that were used
        params = list(lstm.params())
        assert len(params) > 0

        # At least some parameters should have gradients
        params_with_grad = [p for p in params if p.grad is not None]
        assert len(params_with_grad) > 0

    def test_sequence_processing(self):
        """Test processing a sequence and computing loss."""
        lstm = L.LSTM(hidden_size=10, in_size=5)
        linear = L.Linear(1, in_size=10)

        sequence_length = 5
        batch_size = 2

        total_loss = as_variable(np.array(0.0))

        for t in range(sequence_length):
            x = as_variable(np.random.randn(batch_size, 5))
            h = lstm(x)
            y = linear(h)
            target = as_variable(np.random.randn(batch_size, 1))
            loss = F.mean_squared_error(y, target)
            total_loss += loss

        # Backward through entire sequence
        total_loss.backward()

        # All parameters should have gradients
        for param in lstm.params():
            assert param.grad is not None

        for param in linear.params():
            assert param.grad is not None

    def test_params(self):
        """Test parameter counting."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        params = list(lstm.params())

        # LSTM has 4 gates, each with:
        # - x2gate: weight + bias = 2 params
        # - h2gate: weight (no bias) = 1 param
        # Total: 4 * 3 = 12 parameters
        assert len(params) == 12

    def test_cleargrads(self):
        """Test gradient clearing."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        x = as_variable(np.random.randn(2, 5))
        h = lstm(x)
        loss = F.sum(h)
        loss.backward()

        # At least some gradients should exist
        params_with_grad = [p for p in lstm.params() if p.grad is not None]
        assert len(params_with_grad) > 0

        # Clear gradients
        lstm.cleargrads()

        # All gradients should be None after clearing
        for param in lstm.params():
            assert param.grad is None

    def test_batch_sizes(self):
        """Test LSTM with different batch sizes."""
        lstm = L.LSTM(hidden_size=10, in_size=5)

        # Batch size 1
        x1 = as_variable(np.random.randn(1, 5))
        h1 = lstm(x1)
        assert h1.shape == (1, 10)

        lstm.reset_state()

        # Batch size 5
        x5 = as_variable(np.random.randn(5, 5))
        h5 = lstm(x5)
        assert h5.shape == (5, 10)

    def test_hidden_sizes(self):
        """Test LSTM with different hidden sizes."""
        for hidden_size in [1, 10, 50, 100]:
            lstm = L.LSTM(hidden_size=hidden_size, in_size=5)
            x = as_variable(np.random.randn(2, 5))
            h = lstm(x)
            assert h.shape == (2, hidden_size)
            assert lstm.h.shape == (2, hidden_size)
            assert lstm.c.shape == (2, hidden_size)
