import sys
from pathlib import Path

# srcディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from dpl import as_variable
import dpl.layers as L
import dpl.functions as F


def test_sequential_basic():
    """Basic Sequential layer test"""
    print("Testing basic Sequential layer...")

    # Create a simple sequential model
    model = L.Sequential(L.Linear(10), L.Linear(5))

    # Input
    x = as_variable(np.random.randn(3, 20))

    # Forward
    y = model.apply(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (3, 5), f"Output shape mismatch: {y.shape} != (3, 5)"
    print("✓ Basic Sequential test successful")


def test_sequential_with_activation():
    """Sequential layer with activation functions"""
    print("\nTesting Sequential with activation functions...")

    # Sequential with activation functions
    model = L.Sequential(
        L.Linear(128), lambda x: F.relu(x), L.Linear(64), lambda x: F.relu(x), L.Linear(10)
    )

    # Input
    x = as_variable(np.random.randn(5, 20))

    # Forward
    y = model.apply(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (5, 10), f"Output shape mismatch: {y.shape} != (5, 10)"
    print("✓ Sequential with activation test successful")


def test_sequential_backward():
    """Test backward pass through Sequential"""
    print("\nTesting backward pass...")

    model = L.Sequential(L.Linear(10), lambda x: F.relu(x), L.Linear(5))

    x = as_variable(np.random.randn(3, 20))
    y = model.apply(x)

    # Simple loss
    loss = F.sum(y)
    loss.backward()

    # Check gradients exist
    params = list(model.params())
    print(f"  Number of parameters: {len(params)}")

    for param in params:
        assert param.grad is not None, "Gradient should not be None"
        print(f"  Parameter shape: {param.shape}, grad shape: {param.grad.shape}")

    print("✓ Backward pass test successful")


def test_sequential_params():
    """Test parameter tracking in Sequential"""
    print("\nTesting parameter tracking...")

    model = L.Sequential(L.Linear(10), L.Linear(5), L.Linear(3))

    params = list(model.params())
    print(f"  Number of parameters: {len(params)}")

    # Should have 6 parameters: 3 weights + 3 biases
    assert len(params) == 6, f"Expected 6 parameters, got {len(params)}"

    # Initialize by running forward pass first
    x = as_variable(np.random.randn(2, 20))
    _ = model.apply(x)

    for i, param in enumerate(params):
        if param.data is not None:
            print(f"  Parameter {i}: {param.shape}")
        else:
            print(f"  Parameter {i}: not initialized")

    print("✓ Parameter tracking test successful")


def test_sequential_cleargrads():
    """Test cleargrads functionality"""
    print("\nTesting cleargrads...")

    model = L.Sequential(L.Linear(10), L.Linear(5))

    x = as_variable(np.random.randn(3, 20))
    y = model.apply(x)
    loss = F.sum(y)
    loss.backward()

    # Check gradients exist
    for param in model.params():
        assert param.grad is not None, "Gradient should exist after backward"

    # Clear gradients
    model.cleargrads()

    # Check gradients are cleared
    for param in model.params():
        assert param.grad is None, "Gradient should be None after cleargrads"

    print("✓ Cleargrads test successful")


def test_sequential_empty():
    """Test Sequential with single layer"""
    print("\nTesting Sequential with single layer...")

    model = L.Sequential(L.Linear(10))

    x = as_variable(np.random.randn(3, 20))
    y = model.apply(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (3, 10), f"Output shape mismatch: {y.shape} != (3, 10)"
    print("✓ Single layer Sequential test successful")


if __name__ == "__main__":
    print("=" * 50)
    print("Sequential Layer Tests")
    print("=" * 50)

    test_sequential_basic()
    test_sequential_with_activation()
    test_sequential_backward()
    test_sequential_params()
    test_sequential_cleargrads()
    test_sequential_empty()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
