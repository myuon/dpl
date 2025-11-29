import numpy as np

from dpl import Variable
import dpl.functions as F


def test_log_forward():
    """Test log forward pass"""
    x = Variable(np.array([1.0, np.e, np.e**2]))
    y = F.log(x)

    expected = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(y.data, expected, rtol=1e-5)
    print("✓ test_log_forward passed")


def test_log_backward():
    """Test log backward pass"""
    x = Variable(np.array([1.0, 2.0, 4.0]))
    y = F.log(x)
    y.backward()

    # d/dx log(x) = 1/x
    expected_grad = np.array([1.0, 0.5, 0.25])
    np.testing.assert_allclose(x.grad.data, expected_grad, rtol=1e-5)
    print("✓ test_log_backward passed")


def test_log_with_small_values():
    """Test log handles small values without error"""
    x = Variable(np.array([1e-20, 1e-10, 1.0]))
    y = F.log(x)

    # Should not produce NaN or inf
    assert not np.any(np.isnan(y.data))
    assert not np.any(np.isinf(y.data))
    print("✓ test_log_with_small_values passed")


def test_log_gradient_check():
    """Test log gradient with numerical gradient"""
    x_data = np.random.rand(3, 2) + 0.5  # Avoid values close to 0
    x = Variable(x_data)
    y = F.log(x)
    y.backward()

    # Numerical gradient
    eps = 1e-4
    numerical_grad = np.zeros_like(x_data)
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_plus = x_data.copy()
            x_plus[i, j] += eps
            x_minus = x_data.copy()
            x_minus[i, j] -= eps

            y_plus = np.log(np.clip(x_plus, 1e-15, None))
            y_minus = np.log(np.clip(x_minus, 1e-15, None))

            numerical_grad[i, j] = (y_plus[i, j] - y_minus[i, j]) / (2 * eps)

    np.testing.assert_allclose(x.grad.data, numerical_grad, rtol=1e-3, atol=1e-5)
    print("✓ test_log_gradient_check passed")


if __name__ == "__main__":
    test_log_forward()
    test_log_backward()
    test_log_with_small_values()
    test_log_gradient_check()
    print("\n✓ All log tests passed!")
