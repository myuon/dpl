"""Evaluate model on MNIST test set and hard set."""

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


def evaluate(model, X: np.ndarray, y: np.ndarray, batch_size: int = 1000):
    """Evaluate model on dataset.

    Args:
        model: Trained model
        X: Input data (N, 1, 28, 28)
        y: Labels (N,)
        batch_size: Batch size for inference

    Returns:
        Dictionary with evaluation metrics
    """
    all_preds = []
    all_confs = []

    with no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            logits = model(as_variable(X_batch))
            probs = F.softmax(logits).data_required

            all_preds.append(probs.argmax(axis=1))
            all_confs.append(probs.max(axis=1))

    pred = np.concatenate(all_preds)
    conf = np.concatenate(all_confs)

    acc = (pred == y).mean()
    wrong_mask = pred != y

    return {
        "acc": float(acc),
        "mean_conf": float(conf.mean()),
        "wrong_conf": float(conf[wrong_mask].mean()) if wrong_mask.any() else 0.0,
        "num_samples": len(y),
        "num_correct": int((pred == y).sum()),
        "num_wrong": int(wrong_mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MNIST model")
    parser.add_argument(
        "--weights",
        type=str,
        default="mnist_cnn_weights.npz",
        help="Path to model weights",
    )
    parser.add_argument(
        "--hard-set",
        type=str,
        default="hard_set.npz",
        help="Path to hard set (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    print(f"Weights: {args.weights}")
    print("=" * 50)

    # Load model
    print("\nLoading model...")
    model = load_model(args.weights)

    # Evaluate on test set
    print("\nLoading MNIST test set...")
    test_set = MNIST(train=False)
    X_test = test_set.data.reshape(-1, 1, 28, 28)
    y_test = test_set.label

    print("Evaluating on test set...")
    test_results = evaluate(model, X_test, y_test, args.batch_size)

    print("\n" + "=" * 50)
    print("Test Set Results")
    print("=" * 50)
    print(f"Samples: {test_results['num_samples']}")
    print(f"Accuracy: {test_results['acc']:.4f} ({test_results['num_correct']}/{test_results['num_samples']})")
    print(f"Mean confidence: {test_results['mean_conf']:.4f}")
    print(f"Wrong prediction confidence: {test_results['wrong_conf']:.4f}")

    # Evaluate on hard set if available
    try:
        print(f"\nLoading hard set from {args.hard_set}...")
        hard_data = np.load(args.hard_set, allow_pickle=True)
        X_hard = hard_data["X"]
        y_hard = hard_data["y"]

        if len(X_hard) > 0:
            print("Evaluating on hard set...")
            hard_results = evaluate(model, X_hard, y_hard, args.batch_size)

            print("\n" + "=" * 50)
            print("Hard Set Results")
            print("=" * 50)
            print(f"Samples: {hard_results['num_samples']}")
            print(f"Accuracy: {hard_results['acc']:.4f} ({hard_results['num_correct']}/{hard_results['num_samples']})")
            print(f"Mean confidence: {hard_results['mean_conf']:.4f}")
            print(f"Wrong prediction confidence: {hard_results['wrong_conf']:.4f}")

            # Show collection parameters
            if "conf_th" in hard_data:
                print(f"\nHard set collection parameters:")
                print(f"  Confidence threshold: {hard_data['conf_th']}")
                print(f"  Margin threshold: {hard_data['margin_th']}")
        else:
            print("Hard set is empty.")

    except FileNotFoundError:
        print(f"\nHard set not found at {args.hard_set}. Skipping.")
        print("Run collect_hard_set.py to create the hard set.")

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
