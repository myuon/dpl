"""MNIST CNN model definition with weight loading support."""

import dpl.functions as F
import dpl.layers as L


def create_model(hidden_size: int = 1000) -> L.Sequential:
    """Create the MNIST CNN model.

    Architecture:
    - Conv2d(30, 5x5) -> ReLU -> MaxPool(2x2)
    - Flatten
    - Linear(hidden_size) -> ReLU
    - Linear(10)
    """
    model = L.Sequential(
        L.Conv2d(30, kernel_size=5, stride=1, pad=0),
        F.relu,
        lambda x: F.pooling(x, kernel_size=2, stride=2),
        lambda x: x.reshape(x.shape[0], -1),
        L.Linear(hidden_size),
        F.relu,
        L.Linear(10),
    )
    return model


def load_model(weights_path: str, hidden_size: int = 1000) -> L.Sequential:
    """Load the MNIST CNN model with pre-trained weights.

    Args:
        weights_path: Path to the .npz file containing model weights.
        hidden_size: Hidden layer size (must match the saved model).

    Returns:
        The model with loaded weights.
    """
    model = create_model(hidden_size=hidden_size)

    # We need to run a forward pass to initialize the weights before loading
    # This is because Conv2d and Linear layers are lazily initialized
    import numpy as np
    from dpl import as_variable

    dummy_input = np.zeros((1, 1, 28, 28), dtype=np.float32)
    model(as_variable(dummy_input))

    # Now load the weights
    model.load_weights(weights_path)

    return model
