"""Collect hard cases from MNIST dataset after training."""

import argparse

import numpy as np
from sklearn.datasets import fetch_openml

from dpl import as_variable, no_grad, Dataset
import dpl.functions as F

from model import load_model


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


def collect_hard_cases(
    model,
    X: np.ndarray,
    y: np.ndarray,
    conf_th: float = 0.6,
    margin_th: float = 0.1,
):
    """Collect hard cases from the dataset.

    Hard cases are defined as:
    - Misclassified samples
    - Low confidence samples (confidence < conf_th)
    - Low margin samples (margin between top 2 predictions < margin_th)

    Args:
        model: Trained model
        X: Input data (N, 1, 28, 28)
        y: Labels (N,)
        conf_th: Confidence threshold
        margin_th: Margin threshold

    Returns:
        Tuple of (hard_X, hard_y, stats)
    """
    with no_grad():
        logits = model(as_variable(X))
        probs = F.softmax(logits).data_required

    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    mask = (pred != y) | (conf < conf_th) | (margin < margin_th)

    return (
        X[mask],
        y[mask],
        {
            "num": int(mask.sum()),
            "mis": int((pred != y).sum()),
            "low_conf": int((conf < conf_th).sum()),
            "low_margin": int((margin < margin_th).sum()),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Collect hard cases from MNIST")
    parser.add_argument(
        "--weights",
        type=str,
        default="mnist_cnn_weights.npz",
        help="Path to model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hard_set.npz",
        help="Output path for hard set",
    )
    parser.add_argument(
        "--conf-th",
        type=float,
        default=0.6,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--margin-th",
        type=float,
        default=0.1,
        help="Margin threshold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--use-train",
        action="store_true",
        help="Use training set instead of test set",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Collect Hard Cases")
    print("=" * 50)
    print(f"Weights: {args.weights}")
    print(f"Output: {args.output}")
    print(f"Confidence threshold: {args.conf_th}")
    print(f"Margin threshold: {args.margin_th}")
    print(f"Dataset: {'train' if args.use_train else 'test'}")
    print("=" * 50)

    # Load model
    print("\nLoading model...")
    model = load_model(args.weights)

    # Load dataset
    print("Loading MNIST dataset...")
    dataset = MNIST(train=args.use_train)
    print(f"Dataset size: {len(dataset)}")

    # Get all data
    X_all = dataset.data.reshape(-1, 1, 28, 28)
    y_all = dataset.label

    # Collect hard cases in batches
    print("\nCollecting hard cases...")
    hard_X_list = []
    hard_y_list = []
    total_stats = {"num": 0, "mis": 0, "low_conf": 0, "low_margin": 0}

    for i in range(0, len(X_all), args.batch_size):
        X_batch = X_all[i : i + args.batch_size]
        y_batch = y_all[i : i + args.batch_size]

        hard_X, hard_y, stats = collect_hard_cases(
            model, X_batch, y_batch, args.conf_th, args.margin_th
        )

        if len(hard_X) > 0:
            hard_X_list.append(hard_X)
            hard_y_list.append(hard_y)

        for k in total_stats:
            total_stats[k] += stats[k]

        print(f"  Batch {i // args.batch_size + 1}: {stats['num']} hard cases")

    # Concatenate all hard cases
    if hard_X_list:
        hard_X = np.concatenate(hard_X_list, axis=0)
        hard_y = np.concatenate(hard_y_list, axis=0)
    else:
        hard_X = np.array([], dtype=np.float32).reshape(0, 1, 28, 28)
        hard_y = np.array([], dtype=np.int64)

    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics")
    print("=" * 50)
    print(f"Total samples: {len(X_all)}")
    print(
        f"Hard cases: {total_stats['num']} ({100 * total_stats['num'] / len(X_all):.2f}%)"
    )
    print(f"  - Misclassified: {total_stats['mis']}")
    print(f"  - Low confidence: {total_stats['low_conf']}")
    print(f"  - Low margin: {total_stats['low_margin']}")

    # Save hard set
    print(f"\nSaving hard set to {args.output}...")
    np.savez_compressed(
        args.output,
        X=hard_X,
        y=hard_y,
        conf_th=args.conf_th,
        margin_th=args.margin_th,
        stats=total_stats,
    )
    print("Done!")


if __name__ == "__main__":
    main()
