import sys
from pathlib import Path

# srcディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from dpl import as_variable, DataLoader, Trainer
import dpl.functions as F
import dpl.layers as L
import dpl.optimizers as O


def test_trainer_basic():
    """Basic Trainer test with simple data"""
    print("Testing basic Trainer...")

    # Create simple dataset
    np.random.seed(42)
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, size=100).astype(np.int32)

    X_test = np.random.randn(20, 10).astype(np.float32)
    y_test = np.random.randint(0, 2, size=20).astype(np.int32)

    # Create DataLoader
    train_data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    test_data = [(X_test[i], y_test[i]) for i in range(len(X_test))]

    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    # Create model
    model = L.Sequential(L.Linear(5), lambda x: F.relu(x), L.Linear(2))

    # Create optimizer
    optimizer = O.SGD(lr=0.01).setup(model)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=F.softmax_cross_entropy,
        metric_fn=F.accuracy,
        train_loader=train_loader,
        test_loader=test_loader,
        max_epoch=3,
    )

    print(f"  Model: {type(model).__name__}")
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Max epochs: {trainer.max_epoch}")

    # Train
    trainer.run()

    # Check history
    assert len(trainer.train_loss_history) == 3, "Should have 3 train losses"
    assert len(trainer.test_loss_history) == 3, "Should have 3 test losses"
    assert len(trainer.train_metric_history) == 3, "Should have 3 train metrics"
    assert len(trainer.test_metric_history) == 3, "Should have 3 test metrics"

    print(f"  Final train loss: {trainer.train_loss_history[-1]:.4f}")
    print(f"  Final test loss: {trainer.test_loss_history[-1]:.4f}")
    print(f"  Final train metric: {trainer.train_metric_history[-1]:.4f}")
    print(f"  Final test metric: {trainer.test_metric_history[-1]:.4f}")
    print("✓ Basic Trainer test successful")


def test_trainer_step():
    """Test Trainer.step() method"""
    print("\nTesting Trainer.step()...")

    # Create simple dataset
    np.random.seed(42)
    X_train = np.random.randn(50, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, size=50).astype(np.int32)

    train_data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    # Create model and optimizer
    model = L.Sequential(L.Linear(5), lambda x: F.relu(x), L.Linear(2))
    optimizer = O.SGD(lr=0.01).setup(model)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=F.softmax_cross_entropy,
        metric_fn=F.accuracy,
        train_loader=train_loader,
        max_epoch=5,
    )

    # Train for 2 epochs
    trainer.step(epochs=2)
    assert trainer.current_epoch == 2, f"Should be at epoch 2, got {trainer.current_epoch}"
    assert len(trainer.train_loss_history) == 2, "Should have 2 train losses"

    # Train for 1 more epoch
    trainer.step(epochs=1)
    assert trainer.current_epoch == 3, f"Should be at epoch 3, got {trainer.current_epoch}"
    assert len(trainer.train_loss_history) == 3, "Should have 3 train losses"

    print(f"  Current epoch: {trainer.current_epoch}")
    print(f"  History length: {len(trainer.train_loss_history)}")
    print("✓ Trainer.step() test successful")


def test_trainer_with_preprocess():
    """Test Trainer with preprocessing function"""
    print("\nTesting Trainer with preprocessing...")

    # Create simple dataset (images need reshaping)
    np.random.seed(42)
    X_train = np.random.randn(50, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, size=50).astype(np.int32)

    train_data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    # Preprocessing function
    def preprocess(x, t):
        x = x.reshape(-1, 1, 28, 28)  # Reshape to image format
        return x, t

    # Create model
    model = L.Sequential(
        L.Conv2d(16, kernel_size=3, stride=1, pad=1),
        lambda x: F.relu(x),
        lambda x: F.pooling(x, kernel_size=2, stride=2),
        lambda x: x.reshape(x.shape[0], -1),
        L.Linear(10),
    )

    optimizer = O.SGD(lr=0.01).setup(model)

    # Create trainer with preprocessing
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=F.softmax_cross_entropy,
        metric_fn=F.accuracy,
        train_loader=train_loader,
        max_epoch=2,
        preprocess_fn=preprocess,
    )

    # Train
    trainer.run()

    assert len(trainer.train_loss_history) == 2, "Should have 2 train losses"
    print(f"  Final train loss: {trainer.train_loss_history[-1]:.4f}")
    print("✓ Trainer with preprocessing test successful")


def test_trainer_without_metric():
    """Test Trainer without metric function"""
    print("\nTesting Trainer without metric function...")

    # Create simple dataset
    np.random.seed(42)
    X_train = np.random.randn(50, 10).astype(np.float32)
    y_train = np.random.randn(50, 5).astype(np.float32)  # Regression task

    train_data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    # Create model
    model = L.Sequential(L.Linear(5))

    optimizer = O.SGD(lr=0.01).setup(model)

    # Create trainer without metric
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=lambda y, t: F.mean_squared_error(y, t),
        train_loader=train_loader,
        max_epoch=2,
    )

    # Train
    trainer.run()

    assert len(trainer.train_loss_history) == 2, "Should have 2 train losses"
    assert len(trainer.train_metric_history) == 0, "Should have no metrics"
    print(f"  Final train loss: {trainer.train_loss_history[-1]:.4f}")
    print("✓ Trainer without metric test successful")


if __name__ == "__main__":
    print("=" * 50)
    print("Trainer Tests")
    print("=" * 50)

    test_trainer_basic()
    test_trainer_step()
    test_trainer_with_preprocess()
    test_trainer_without_metric()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
