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
        b_h = rnn.x2h.b.data  # shape: (hidden_size,)
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

    def test_rnn_bptt_gradients(self):
        """Test RNN BPTT gradients against numerical differentiation.

        Test case: 2-step RNN with simple loss
        h0 = 0
        h1 = tanh(W_xh @ x1 + W_hh @ h0 + b)
        h2 = tanh(W_xh @ x2 + W_hh @ h1 + b)
        loss = sum(h2)

        Verify that gradients computed by backprop match numerical gradients.
        """
        np.random.seed(42)
        hidden_size = 3
        in_size = 2
        batch_size = 1  # Use batch_size=1 for simpler numerical diff

        # Create RNN layer
        rnn = L.RNN(hidden_size=hidden_size, in_size=in_size)

        # Create fixed inputs
        x1 = np.random.randn(batch_size, in_size)
        x2 = np.random.randn(batch_size, in_size)

        # Helper function to compute loss given current parameters
        def compute_loss():
            rnn.reset_state()
            h1 = rnn(as_variable(x1))
            h2 = rnn(as_variable(x2))
            loss = F.sum(h2)
            return loss

        # Compute gradients with backprop
        rnn.cleargrads()
        loss = compute_loss()
        loss.backward()
        loss.unchain_backward()

        # Save backprop gradients
        grad_W_xh_backprop = rnn.x2h.W.grad.data.copy()
        grad_b_backprop = rnn.x2h.b.grad.data.copy()
        grad_W_hh_backprop = rnn.h2h.W.grad.data.copy()

        # Numerical gradient computation
        eps = 1e-4

        def numerical_gradient(param_array):
            """Compute numerical gradient for a parameter array."""
            grad = np.zeros_like(param_array)
            it = np.nditer(param_array, flags=["multi_index"])

            while not it.finished:
                idx = it.multi_index
                original_value = param_array[idx]

                # f(x + h)
                param_array[idx] = original_value + eps
                rnn.cleargrads()  # Clear gradients before forward
                loss_plus = compute_loss()
                loss_plus_val = float(loss_plus.data)

                # f(x - h)
                param_array[idx] = original_value - eps
                rnn.cleargrads()  # Clear gradients before forward
                loss_minus = compute_loss()
                loss_minus_val = float(loss_minus.data)

                # Numerical gradient
                grad[idx] = (loss_plus_val - loss_minus_val) / (2 * eps)

                # Restore original value
                param_array[idx] = original_value

                it.iternext()

            return grad

        # Compute numerical gradients for each parameter
        grad_W_xh_numerical = numerical_gradient(rnn.x2h.W.data)
        grad_b_numerical = numerical_gradient(rnn.x2h.b.data)
        grad_W_hh_numerical = numerical_gradient(rnn.h2h.W.data)

        # Compare gradients
        print("\n=== BPTT Gradient Check ===")
        print(f"W_xh backprop:\n{grad_W_xh_backprop}")
        print(f"W_xh numerical:\n{grad_W_xh_numerical}")
        print(
            f"W_xh gradient difference: {np.max(np.abs(grad_W_xh_backprop - grad_W_xh_numerical))}"
        )
        print(f"\nb backprop:\n{grad_b_backprop}")
        print(f"b numerical:\n{grad_b_numerical}")
        print(
            f"b gradient difference: {np.max(np.abs(grad_b_backprop - grad_b_numerical))}"
        )
        print(f"\nW_hh backprop:\n{grad_W_hh_backprop}")
        print(f"W_hh numerical:\n{grad_W_hh_numerical}")
        print(
            f"W_hh gradient difference: {np.max(np.abs(grad_W_hh_backprop - grad_W_hh_numerical))}"
        )

        # Assertions
        assert np.allclose(
            grad_W_xh_backprop, grad_W_xh_numerical, rtol=1e-3, atol=1e-5
        ), "W_xh gradient mismatch"
        assert np.allclose(
            grad_b_backprop, grad_b_numerical, rtol=1e-3, atol=1e-5
        ), "b gradient mismatch"
        assert np.allclose(
            grad_W_hh_backprop, grad_W_hh_numerical, rtol=1e-3, atol=1e-5
        ), "W_hh gradient mismatch"

    def test_rnn_bptt_longer_sequence(self):
        """Test RNN BPTT with a longer sequence (5 steps)."""
        np.random.seed(123)
        hidden_size = 4
        in_size = 3
        batch_size = 1
        num_steps = 5

        # Create RNN layer
        rnn = L.RNN(hidden_size=hidden_size, in_size=in_size)

        # Create fixed inputs
        xs = [np.random.randn(batch_size, in_size) for _ in range(num_steps)]

        def compute_loss():
            rnn.reset_state()
            for x in xs:
                h = rnn(as_variable(x))
            loss = F.sum(h)  # Loss from final hidden state
            return loss

        # Compute gradients with backprop
        rnn.cleargrads()
        loss = compute_loss()
        loss.backward()
        loss.unchain_backward()

        # Save backprop gradients
        grad_W_hh_backprop = rnn.h2h.W.grad.data.copy()

        # Numerical gradient for W_hh (just check one parameter for speed)
        eps = 1e-4

        def numerical_gradient_single_param(param_array, i, j):
            """Compute numerical gradient for a single parameter."""
            original_value = param_array[i, j]

            param_array[i, j] = original_value + eps
            loss_plus = compute_loss()
            loss_plus_val = float(loss_plus.data)

            param_array[i, j] = original_value - eps
            loss_minus = compute_loss()
            loss_minus_val = float(loss_minus.data)

            grad = (loss_plus_val - loss_minus_val) / (2 * eps)

            param_array[i, j] = original_value

            return grad

        # Check a few random elements of W_hh
        for _ in range(5):
            i = np.random.randint(0, hidden_size)
            j = np.random.randint(0, hidden_size)

            grad_numerical = numerical_gradient_single_param(rnn.h2h.W.data, i, j)
            grad_backprop = grad_W_hh_backprop[i, j]

            assert np.isclose(
                grad_backprop, grad_numerical, rtol=1e-3, atol=1e-5
            ), f"W_hh[{i},{j}] gradient mismatch: {grad_backprop} vs {grad_numerical}"
