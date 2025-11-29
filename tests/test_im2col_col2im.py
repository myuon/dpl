import numpy as np

from dpl import Variable
import dpl.functions as F


def test_im2col_basic():
    """Test basic im2col functionality"""
    # Simple input: 1 batch, 1 channel, 4x4 image
    x_data = np.arange(16).reshape(1, 1, 4, 4).astype(np.float32)
    x = Variable(x_data)

    # Apply im2col with 2x2 filter, stride=1, pad=0
    col = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=0)

    # Output shape should be (9, 4)
    # 9 = number of windows (3x3), 4 = filter size (2x2)
    assert col.shape == (9, 4), f"Expected shape (9, 4), got {col.shape}"

    print("✓ test_im2col_basic passed")


def test_im2col_with_padding():
    """Test im2col with padding"""
    x_data = np.arange(16).reshape(1, 1, 4, 4).astype(np.float32)
    x = Variable(x_data)

    # Apply im2col with padding
    col = F.im2col(x, filter_h=3, filter_w=3, stride=1, pad=1)

    # Output shape should be (16, 9)
    # 16 = number of windows (4x4 with padding), 9 = filter size (3x3)
    assert col.shape == (16, 9), f"Expected shape (16, 9), got {col.shape}"

    print("✓ test_im2col_with_padding passed")


def test_col2im_basic():
    """Test basic col2im functionality"""
    # Create a simple input
    x_data = np.arange(16).reshape(1, 1, 4, 4).astype(np.float32)
    x = Variable(x_data)

    # Apply im2col
    col = F.im2col(x, filter_h=2, filter_w=2, stride=2, pad=0)

    # Apply col2im to reconstruct
    img = F.col2im(col, input_shape=(1, 1, 4, 4), filter_h=2, filter_w=2, stride=2, pad=0)

    # Check shape
    assert img.shape == x.shape, f"Expected shape {x.shape}, got {img.shape}"

    # With non-overlapping windows (stride=2), reconstruction should be exact
    np.testing.assert_allclose(img.data, x_data, rtol=1e-5,
                               err_msg=f"Expected:\n{x_data}\nGot:\n{img.data}")

    print("✓ test_col2im_basic passed")


def test_im2col_col2im_with_stride():
    """Test im2col and col2im with stride=1 (overlapping windows)"""
    x_data = np.arange(9).reshape(1, 1, 3, 3).astype(np.float32)
    x = Variable(x_data)

    # Apply im2col with stride=1
    col = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=0)

    # Apply col2im
    img = F.col2im(col, input_shape=(1, 1, 3, 3), filter_h=2, filter_w=2, stride=1, pad=0)

    # Check shape
    assert img.shape == x.shape, f"Expected shape {x.shape}, got {img.shape}"

    # With overlapping windows, values will be summed
    # Each pixel appears in multiple windows, so values will be larger
    # The reconstruction won't be exact, but the operation should be consistent
    print(f"Original:\n{x_data[0, 0]}")
    print(f"Reconstructed (with overlaps):\n{img.data[0, 0]}")

    print("✓ test_im2col_col2im_with_stride passed")


def test_col2im_gradient_accumulation():
    """Test that col2im properly accumulates gradients from overlapping regions"""
    # Simple 3x3 input
    x_data = np.ones((1, 1, 3, 3), dtype=np.float32)
    x = Variable(x_data)

    # im2col with stride=1 creates overlapping windows
    col = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=0)

    # Create gradient with all ones
    dcol = Variable(np.ones_like(col.data))

    # col2im should accumulate gradients
    dx = F.col2im(dcol, input_shape=(1, 1, 3, 3), filter_h=2, filter_w=2, stride=1, pad=0)

    # Expected gradient accumulation pattern:
    # Corner pixels appear in 1 window: 1
    # Edge pixels appear in 2 windows: 2
    # Center pixel appears in 4 windows: 4
    expected = np.array([[[[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]]], dtype=np.float32)

    np.testing.assert_allclose(dx.data, expected, rtol=1e-5,
                               err_msg=f"Expected:\n{expected[0,0]}\nGot:\n{dx.data[0,0]}")

    print("✓ test_col2im_gradient_accumulation passed")


def test_col2im_with_padding():
    """Test col2im with padding"""
    x_data = np.arange(16).reshape(1, 1, 4, 4).astype(np.float32)
    x = Variable(x_data)

    # im2col with padding
    col = F.im2col(x, filter_h=2, filter_w=2, stride=1, pad=1)

    # col2im with same parameters
    img = F.col2im(col, input_shape=(1, 1, 4, 4), filter_h=2, filter_w=2, stride=1, pad=1)

    # Check shape
    assert img.shape == x.shape, f"Expected shape {x.shape}, got {img.shape}"

    print("✓ test_col2im_with_padding passed")


def test_im2col_col2im_multichannel():
    """Test im2col and col2im with multiple channels"""
    # 2 batches, 3 channels, 4x4 image
    x_data = np.random.randn(2, 3, 4, 4).astype(np.float32)
    x = Variable(x_data)

    # im2col
    col = F.im2col(x, filter_h=2, filter_w=2, stride=2, pad=0)

    # Expected shape: (2*2*2, 3*2*2) = (8, 12)
    # 8 = N * out_h * out_w = 2 * 2 * 2
    # 12 = C * filter_h * filter_w = 3 * 2 * 2
    assert col.shape == (8, 12), f"Expected shape (8, 12), got {col.shape}"

    # col2im
    img = F.col2im(col, input_shape=(2, 3, 4, 4), filter_h=2, filter_w=2, stride=2, pad=0)

    # Check shape
    assert img.shape == x.shape, f"Expected shape {x.shape}, got {img.shape}"

    # With non-overlapping windows, should match
    np.testing.assert_allclose(img.data, x_data, rtol=1e-5)

    print("✓ test_im2col_col2im_multichannel passed")


def test_col2im_with_variable():
    """Test col2im with Variable input"""
    x_data = np.arange(16).reshape(1, 1, 4, 4).astype(np.float32)
    x = Variable(x_data)

    # im2col returns Variable
    col = F.im2col(x, filter_h=2, filter_w=2, stride=2, pad=0)

    # col2im should accept Variable
    img = F.col2im(col, input_shape=(1, 1, 4, 4), filter_h=2, filter_w=2, stride=2, pad=0)

    # Check that it works
    assert img.shape == x.shape
    np.testing.assert_allclose(img.data, x_data, rtol=1e-5)

    print("✓ test_col2im_with_variable passed")


if __name__ == "__main__":
    test_im2col_basic()
    test_im2col_with_padding()
    test_col2im_basic()
    test_im2col_col2im_with_stride()
    test_col2im_gradient_accumulation()
    test_col2im_with_padding()
    test_im2col_col2im_multichannel()
    test_col2im_with_variable()
    print("\n✓ All im2col/col2im tests passed!")
