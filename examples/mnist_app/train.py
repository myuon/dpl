"""Train MNIST CNN model and save weights."""

import argparse

import numpy as np
from sklearn.datasets import fetch_openml

import dpl
import dpl.functions as F
import dpl.optimizers as O
from dpl import DataLoader, Dataset, Trainer

from model import create_model


class MNIST(Dataset):
    """MNIST dataset loaded from OpenML."""

    def prepare(self):
        mnist = fetch_openml("mnist_784", version=1)
        data = mnist.data.values.astype(np.float32) / 255.0
        label = mnist.target.values.astype(int)

        if self.train:
            self.data = data[:60000]
            self.label = label[:60000]
        else:
            self.data = data[60000:]
            self.label = label[60000:]


def main():
    parser = argparse.ArgumentParser(description="Train MNIST CNN model")
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--hidden-size", type=int, default=1000, help="Hidden layer size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mnist_cnn_weights.npz",
        help="Output path for model weights",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=True, help="Use GPU if available"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("MNIST CNN Training")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Output: {args.output}")
    print(f"GPU: {args.gpu and dpl.metal.gpu_enable}")
    print("=" * 50)

    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    train_set = MNIST(train=True)
    test_set = MNIST(train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")

    # Create model and optimizer
    print("\nCreating model...")
    model = create_model(hidden_size=args.hidden_size)
    optimizer = O.Adam(lr=args.lr).setup(model)
    optimizer.add_hook(O.WeightDecay(1e-4))

    if args.gpu and dpl.metal.gpu_enable:
        print("Using GPU...")
        model.to_gpu()
        train_loader.to_gpu()
        test_loader.to_gpu()

    # Preprocessing function to reshape images for CNN
    def preprocess(x, t):
        # Reshape input to (N, C, H, W) format for CNN
        x = x.reshape(-1, 1, 28, 28)
        return x, t

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=F.softmax_cross_entropy,
        metric_fn=F.accuracy,
        train_loader=train_loader,
        test_loader=test_loader,
        max_epoch=args.epochs,
        preprocess_fn=preprocess,
        max_grad=25.0,
        clip_grads=True,
    )

    # Train the model
    print("\nTraining...")
    trainer.run()

    # Save weights
    print(f"\nSaving weights to {args.output}...")
    model.save_weights(args.output)
    print("Done!")

    # Print final metrics
    print("\n" + "=" * 50)
    print("Final Results")
    print("=" * 50)
    print(f"Final train loss: {trainer.train_loss_history[-1]:.4f}")
    print(f"Final test loss: {trainer.test_loss_history[-1]:.4f}")
    print(f"Final train accuracy: {trainer.train_metric_history[-1]:.4f}")
    print(f"Final test accuracy: {trainer.test_metric_history[-1]:.4f}")


if __name__ == "__main__":
    main()
