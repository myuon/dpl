import numpy as np

from dpl import Variable
import dpl.functions as F


def test_im2col_backward():
    """Test im2col backward pass"""
    # Simple input
    x_data = np.random.randn(1, 1, 4, 4).astype(np.float32)
    x = Variable(x_data)

    # Forward: im2col
    col = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=0)

    # Backward: sum and compute gradients
    loss = F.sum(col)
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "x.grad should not be None after backward"
    assert x.grad.shape == x.shape, f"x.grad shape mismatch: {x.grad.shape} vs {x.shape}"

    # Gradient should not be all zeros
    assert not np.allclose(x.grad.data, 0), "x.grad should not be all zeros"

    print("✓ test_im2col_backward passed")


def test_col2im_backward():
    """Test col2im backward pass"""
    # Simple column data
    col_data = np.random.randn(9, 4).astype(np.float32)
    col = Variable(col_data)

    # Forward: col2im
    img = F.col2im(col, input_shape=(1, 1, 4, 4), filter_h=2, filter_w=2, stride=1, pad=0)

    # Backward: sum and compute gradients
    loss = F.sum(img)
    loss.backward()

    # Check that gradients exist
    assert col.grad is not None, "col.grad should not be None after backward"
    assert col.grad.shape == col.shape, f"col.grad shape mismatch: {col.grad.shape} vs {col.shape}"

    # Gradient should not be all zeros
    assert not np.allclose(col.grad.data, 0), "col.grad should not be all zeros"

    print("✓ test_col2im_backward passed")


def test_im2col_col2im_backward_chain():
    """Test im2col -> col2im backward chain"""
    # Simple input
    x_data = np.random.randn(1, 1, 4, 4).astype(np.float32)
    x = Variable(x_data)

    # Forward: im2col -> col2im (should be identity-like for non-overlapping)
    col = F.im2col(x, filter_h=2, filter_w=2, stride=2, pad=0)
    img = F.col2im(col, input_shape=(1, 1, 4, 4), filter_h=2, filter_w=2, stride=2, pad=0)

    # Backward
    loss = F.sum(img)
    loss.backward()

    # Check that gradients flow through the chain
    assert x.grad is not None, "x.grad should not be None"
    assert x.grad.shape == x.shape, f"x.grad shape mismatch"

    # For non-overlapping windows (stride=2), gradient should be all ones
    # because im2col -> col2im is identity-like
    expected_grad = np.ones_like(x_data)
    np.testing.assert_allclose(x.grad.data, expected_grad, rtol=1e-5,
                               err_msg=f"Expected:\n{expected_grad}\nGot:\n{x.grad.data}")

    print("✓ test_im2col_col2im_backward_chain passed")


def test_im2col_backward_with_overlap():
    """Test im2col backward with overlapping windows"""
    # 3x3 input
    x_data = np.ones((1, 1, 3, 3), dtype=np.float32)
    x = Variable(x_data)

    # im2col with stride=1 (overlapping)
    col = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=0)

    # Backward with all-ones gradient
    loss = F.sum(col)
    loss.backward()

    # Expected gradient pattern:
    # Corner pixels appear in 1 window: 1
    # Edge pixels appear in 2 windows: 2
    # Center pixel appears in 4 windows: 4
    expected_grad = np.array([[[[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]]], dtype=np.float32)

    assert x.grad is not None
    np.testing.assert_allclose(x.grad.data, expected_grad, rtol=1e-5,
                               err_msg=f"Expected:\n{expected_grad[0,0]}\nGot:\n{x.grad.data[0,0]}")

    print("✓ test_im2col_backward_with_overlap passed")


def test_gradient_numerical_check_im2col():
    """Numerical gradient check for im2col"""
    np.random.seed(42)
    x_data = np.random.randn(1, 1, 3, 3).astype(np.float32)

    # Compute gradient via backprop
    x = Variable(x_data.copy())
    col = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=0)
    loss = F.sum(col)
    loss.backward()

    # Numerical gradient
    def f(x_input):
        x_var = Variable(x_input)
        col_var = F.im2col(x_var, filter_h=2, filter_w=2, stride=1, pad=0)
        return np.sum(col_var.data)

    eps = 1e-4
    numerical_grad = np.zeros_like(x_data)
    for i in range(x_data.shape[2]):
        for j in range(x_data.shape[3]):
            x_plus = x_data.copy()
            x_plus[0, 0, i, j] += eps
            x_minus = x_data.copy()
            x_minus[0, 0, i, j] -= eps

            numerical_grad[0, 0, i, j] = (f(x_plus) - f(x_minus)) / (2 * eps)

    # Compare
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.data, numerical_grad, rtol=1e-3, atol=1e-4,
                               err_msg="Gradient mismatch between backprop and numerical")

    print("✓ test_gradient_numerical_check_im2col passed")


def test_gradient_numerical_check_col2im():
    """Numerical gradient check for col2im"""
    np.random.seed(42)
    col_data = np.random.randn(4, 4).astype(np.float32)  # 2x2 windows from 3x3 image

    # Compute gradient via backprop
    col = Variable(col_data.copy())
    img = F.col2im(col, input_shape=(1, 1, 3, 3), filter_h=2, filter_w=2, stride=1, pad=0)
    loss = F.sum(img)
    loss.backward()

    # Numerical gradient
    def f(col_input):
        col_var = Variable(col_input)
        img_var = F.col2im(col_var, input_shape=(1, 1, 3, 3), filter_h=2, filter_w=2, stride=1, pad=0)
        return np.sum(img_var.data)

    eps = 1e-4
    numerical_grad = np.zeros_like(col_data)
    for i in range(col_data.shape[0]):
        for j in range(col_data.shape[1]):
            col_plus = col_data.copy()
            col_plus[i, j] += eps
            col_minus = col_data.copy()
            col_minus[i, j] -= eps

            numerical_grad[i, j] = (f(col_plus) - f(col_minus)) / (2 * eps)

    # Compare
    assert col.grad is not None
    np.testing.assert_allclose(col.grad.data, numerical_grad, rtol=1e-3, atol=1e-4,
                               err_msg="Gradient mismatch between backprop and numerical")

    print("✓ test_gradient_numerical_check_col2im passed")


if __name__ == "__main__":
    test_im2col_backward()
    test_col2im_backward()
    test_im2col_col2im_backward_chain()
    test_im2col_backward_with_overlap()
    test_gradient_numerical_check_im2col()
    test_gradient_numerical_check_col2im()
    print("\n✓ All im2col/col2im backward tests passed!")
