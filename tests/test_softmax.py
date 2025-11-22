import numpy as np
import sys

sys.path.insert(0, "../src")

from dpl import Variable
import dpl.functions as F


def test_softmax_forward():
    """Test softmax forward pass"""
    x = Variable(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]))
    y = F.softmax(x)

    # Check that each row sums to 1
    row_sums = np.sum(y.data, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(2), rtol=1e-5)

    # Check that all values are positive
    assert np.all(y.data > 0)
    assert np.all(y.data < 1)

    print("✓ test_softmax_forward passed")


def test_softmax_backward():
    """Test softmax backward pass"""
    x = Variable(np.array([[1.0, 2.0, 3.0]]))
    y = F.softmax(x)
    loss = F.sum(y)
    loss.backward()

    # Gradient should exist
    assert x.grad is not None
    # Just check that gradient is computed
    assert x.grad.shape == x.shape

    print("✓ test_softmax_backward passed")


def test_softmax_cross_entropy_forward():
    """Test softmax cross entropy forward pass"""
    x = Variable(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]]))
    t = Variable(np.array([2, 0]))  # Correct labels

    loss = F.softmax_cross_entropy(x, t)

    # Loss should be positive
    assert loss.data > 0

    # Loss should be approximately -log(p) where p is probability of correct class
    p = F.softmax(x)
    expected_loss = -(np.log(p.data[0, 2]) + np.log(p.data[1, 0])) / 2
    np.testing.assert_allclose(loss.data, expected_loss, rtol=1e-5)

    print("✓ test_softmax_cross_entropy_forward passed")


def test_softmax_cross_entropy_backward():
    """Test softmax cross entropy backward pass - THIS IS THE KEY TEST"""
    x = Variable(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]]))
    t = Variable(np.array([2, 0]))

    loss = F.softmax_cross_entropy(x, t)
    loss.backward()

    # CRITICAL: x.grad should NOT be None
    assert x.grad is not None, "Gradient is None! The computational graph is broken!"

    # Check gradient shape
    assert x.grad.shape == x.shape

    # Gradient should be non-zero
    assert not np.allclose(x.grad.data, 0)

    print("✓ test_softmax_cross_entropy_backward passed")


def test_softmax_cross_entropy_gradient_check():
    """Test softmax cross entropy gradient with numerical gradient"""
    x_data = np.random.randn(3, 4)
    t_data = np.array([1, 0, 3])

    x = Variable(x_data)
    t = Variable(t_data)

    loss = F.softmax_cross_entropy(x, t)
    loss.backward()

    # Numerical gradient
    eps = 1e-4
    numerical_grad = np.zeros_like(x_data)
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_plus = x_data.copy()
            x_plus[i, j] += eps
            x_minus = x_data.copy()
            x_minus[i, j] -= eps

            # Forward pass for numerical gradient
            def forward(x_input):
                N = x_input.shape[0]
                # Softmax
                exp_x = np.exp(x_input - np.max(x_input, axis=1, keepdims=True))
                p = exp_x / np.sum(exp_x, axis=1, keepdims=True)
                # Cross entropy
                log_p = np.log(np.clip(p, 1e-15, 1.0))
                tlog_p = log_p[np.arange(N), t_data]
                return -np.sum(tlog_p) / N

            loss_plus = forward(x_plus)
            loss_minus = forward(x_minus)

            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(x.grad.data, numerical_grad, rtol=1e-3, atol=1e-5)
    print("✓ test_softmax_cross_entropy_gradient_check passed")


def test_softmax_cross_entropy_perfect_prediction():
    """Test with very confident correct predictions"""
    # Very high score for correct class
    x = Variable(np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]))
    t = Variable(np.array([0, 1]))

    loss = F.softmax_cross_entropy(x, t)

    # Loss should be very small for perfect predictions
    assert loss.data < 0.1

    print("✓ test_softmax_cross_entropy_perfect_prediction passed")


def test_softmax_cross_entropy_wrong_prediction():
    """Test with very confident wrong predictions"""
    # Very high score for wrong class
    x = Variable(np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]))
    t = Variable(np.array([0, 1]))

    loss = F.softmax_cross_entropy(x, t)

    # Loss should be very large for wrong predictions
    assert loss.data > 5.0

    print("✓ test_softmax_cross_entropy_wrong_prediction passed")


if __name__ == "__main__":
    test_softmax_forward()
    test_softmax_backward()
    test_softmax_cross_entropy_forward()
    test_softmax_cross_entropy_backward()
    test_softmax_cross_entropy_gradient_check()
    test_softmax_cross_entropy_perfect_prediction()
    test_softmax_cross_entropy_wrong_prediction()
    print("\n✓ All softmax tests passed!")
