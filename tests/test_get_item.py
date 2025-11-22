import numpy as np
import sys
sys.path.insert(0, '/Users/ioijoi/ghq/github.com/myuon/dpl/src')

from dpl import Variable
import dpl.functions as F


def test_get_item_1d():
    """Test get_item with 1D indexing"""
    x = Variable(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    y = x[2]

    assert y.data == 3.0
    print("✓ test_get_item_1d passed")


def test_get_item_2d():
    """Test get_item with 2D indexing"""
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = x[1, 2]

    assert y.data == 6.0
    print("✓ test_get_item_2d passed")


def test_get_item_slice():
    """Test get_item with slicing"""
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = x[0, :]

    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y.data, expected)
    print("✓ test_get_item_slice passed")


def test_get_item_advanced_indexing():
    """Test get_item with advanced indexing"""
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    indices = np.array([0, 2, 1])
    cols = np.array([1, 2, 0])
    y = x[indices, cols]

    expected = np.array([2.0, 9.0, 4.0])
    np.testing.assert_array_equal(y.data, expected)
    print("✓ test_get_item_advanced_indexing passed")


def test_get_item_backward():
    """Test get_item backward pass"""
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = x[1, 2]
    y.backward()

    expected_grad = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(x.grad.data, expected_grad)
    print("✓ test_get_item_backward passed")


def test_get_item_advanced_backward():
    """Test get_item backward with advanced indexing"""
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    indices = np.array([0, 1])
    cols = np.array([2, 0])
    y = x[indices, cols]  # Should give [3.0, 4.0]
    loss = F.sum(y)
    loss.backward()

    expected_grad = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(x.grad.data, expected_grad)
    print("✓ test_get_item_advanced_backward passed")


def test_get_item_gradient_check():
    """Test get_item gradient with numerical gradient"""
    x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    x = Variable(x_data)

    indices = np.array([0, 1, 2])
    cols = np.array([1, 2, 0])
    y = x[indices, cols]
    loss = F.sum(y * y)  # Use a more complex function
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

            y_plus = x_plus[indices, cols]
            y_minus = x_minus[indices, cols]

            loss_plus = np.sum(y_plus * y_plus)
            loss_minus = np.sum(y_minus * y_minus)

            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(x.grad.data, numerical_grad, rtol=1e-3, atol=1e-5)
    print("✓ test_get_item_gradient_check passed")


if __name__ == "__main__":
    test_get_item_1d()
    test_get_item_2d()
    test_get_item_slice()
    test_get_item_advanced_indexing()
    test_get_item_backward()
    test_get_item_advanced_backward()
    test_get_item_gradient_check()
    print("\n✓ All get_item tests passed!")
