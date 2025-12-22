from typing import Callable, Optional, Iterable, Any
import time
from dpl.core import Variable, as_variable, no_grad, ndarray
from dpl.layers import Layer, StatefulLayer
from dpl.optimizers import Optimizer
from dpl import functions as F


class Trainer:
    def __init__(
        self,
        model: Layer,
        optimizer: Optimizer,
        loss_fn: Callable[[Variable, Variable], Variable],
        metric_fn: Optional[Callable[[Variable, Variable], Variable]] = None,
        train_loader: Optional[Any] = None,
        test_loader: Optional[Any] = None,
        max_epoch: int = 10,
        preprocess_fn: Optional[
            Callable[[ndarray, ndarray], tuple[ndarray, ndarray]]
        ] = None,
        train_preprocess_fn: Optional[
            Callable[[ndarray, ndarray], tuple[ndarray, ndarray]]
        ] = None,
        eval_preprocess_fn: Optional[
            Callable[[ndarray, ndarray], tuple[ndarray, ndarray]]
        ] = None,
        on_epoch_start: Optional[Callable[["Trainer"], None]] = None,
        on_epoch_end: Optional[Callable[["Trainer"], None]] = None,
        on_batch_end: Optional[Callable[["Trainer"], None]] = None,
        truncate_bptt: bool = False,
        clip_grads: bool = False,
        max_grad: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epoch = max_epoch
        self.preprocess_fn = preprocess_fn
        self.train_preprocess_fn = train_preprocess_fn
        self.eval_preprocess_fn = eval_preprocess_fn
        self.truncate_bptt = truncate_bptt
        self.clip_grads = clip_grads
        self.max_grad = max_grad

        # Callbacks
        self.on_epoch_start = on_epoch_start
        self.on_epoch_end = on_epoch_end
        self.on_batch_end = on_batch_end

        # History tracking
        self.train_loss_history: list[float] = []
        self.test_loss_history: list[float] = []
        self.train_metric_history: list[float] = []
        self.test_metric_history: list[float] = []
        self.epoch_times: list[float] = []

        # Current state
        self.current_epoch = 0
        self.current_batch = 0
        self.total_time = 0.0

    def _clip_gradients(self) -> None:
        import numpy as np

        """Clip gradients by global norm."""
        # Calculate global norm
        total_norm = 0.0
        for param in self.model.params():
            if param.grad is not None:
                param_norm = F.sum(param.grad**2)
                total_norm += param_norm.data_required

        total_norm = np.sqrt(total_norm)

        # Clip gradients if norm exceeds max_grad
        if total_norm > self.max_grad:
            clip_coef = self.max_grad / (total_norm + 1e-6)
            for param in self.model.params():
                if param.grad is not None:
                    param.grad *= clip_coef

    def train_epoch(self) -> tuple[float, Optional[float]]:
        """Train for one epoch.

        Returns:
            Tuple of (average_loss, average_metric)
        """
        if self.train_loader is None:
            raise ValueError("train_loader must be provided to train")

        sum_loss = 0.0
        sum_metric = 0.0
        total_samples = 0
        batch_count = 0

        for x, t in self.train_loader:
            # Preprocess if needed (train_preprocess_fn takes priority)
            preprocess = self.train_preprocess_fn or self.preprocess_fn
            if preprocess is not None:
                x, t = preprocess(x, t)

            # Convert to Variable
            x, t = as_variable(x), as_variable(t)

            # Forward
            y = self.model(x)
            loss = self.loss_fn(y, t)

            # Backward
            loss.backward()

            # Truncate computational graph if using truncated BPTT
            if self.truncate_bptt:
                loss.unchain_backward()

            # Clip gradients if enabled
            if self.clip_grads:
                self._clip_gradients()

            self.optimizer.update()
            self.model.cleargrads()

            # Track metrics
            batch_size = len(t.data_required)
            sum_loss += loss.data_required.astype(float).item() * batch_size
            total_samples += batch_size
            batch_count += 1

            if self.metric_fn is not None:
                metric = self.metric_fn(y, t)
                sum_metric += metric.data_required.astype(float).item() * batch_size

            # Batch callback
            self.current_batch = batch_count
            if self.on_batch_end is not None:
                self.on_batch_end(self)

        avg_loss = sum_loss / total_samples
        avg_metric = sum_metric / total_samples if self.metric_fn is not None else None

        return avg_loss, avg_metric

    def test_epoch(self) -> tuple[float, Optional[float]]:
        """Evaluate on test/validation set.

        Returns:
            Tuple of (average_loss, average_metric)
        """
        if self.test_loader is None:
            raise ValueError("test_loader must be provided to test")

        sum_loss = 0.0
        sum_metric = 0.0
        total_samples = 0

        with no_grad():
            for x, t in self.test_loader:
                # Preprocess if needed (eval_preprocess_fn takes priority)
                preprocess = self.eval_preprocess_fn or self.preprocess_fn
                if preprocess is not None:
                    x, t = preprocess(x, t)

                # Convert to Variable
                x, t = as_variable(x), as_variable(t)

                # Forward
                y = self.model(x)
                loss = self.loss_fn(y, t)

                # Track metrics
                batch_size = len(t.data_required)
                sum_loss += loss.data_required.astype(float).item() * batch_size
                total_samples += batch_size

                if self.metric_fn is not None:
                    metric = self.metric_fn(y, t)
                    sum_metric += metric.data_required.astype(float).item() * batch_size

        avg_loss = sum_loss / total_samples
        avg_metric = sum_metric / total_samples if self.metric_fn is not None else None

        return avg_loss, avg_metric

    def step(self, epochs: int = 1) -> None:
        """Train for specified number of epochs.

        Args:
            epochs: Number of epochs to train
        """
        for _ in range(epochs):
            # Epoch start callback
            if self.on_epoch_start is not None:
                self.on_epoch_start(self)

            epoch_start = time.time()

            # Train
            train_loss, train_metric = self.train_epoch()
            self.train_loss_history.append(train_loss)
            if train_metric is not None:
                self.train_metric_history.append(train_metric)

            # Test
            test_loss = None
            test_metric = None
            if self.test_loader is not None:
                test_loss, test_metric = self.test_epoch()
                self.test_loss_history.append(test_loss)
                if test_metric is not None:
                    self.test_metric_history.append(test_metric)

            # Track time
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            self.total_time += epoch_time

            # Update state
            self.current_epoch += 1

            # Epoch end callback
            if self.on_epoch_end is not None:
                self.on_epoch_end(self)

            # Print progress
            self._print_progress(
                train_loss, train_metric, test_loss, test_metric, epoch_time
            )

    def _print_progress(
        self,
        train_loss: float,
        train_metric: Optional[float],
        test_loss: Optional[float],
        test_metric: Optional[float],
        epoch_time: float,
    ) -> None:
        """Print training progress."""
        # Calculate ETA
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.max_epoch - self.current_epoch
        estimated_remaining = avg_epoch_time * remaining_epochs

        # Build message
        msg = f"epoch {self.current_epoch}/{self.max_epoch}, "
        msg += f"loss: {train_loss:.4f}"

        if train_metric is not None:
            msg += f", metric: {train_metric:.4f}"

        if test_loss is not None:
            msg += f", test_loss: {test_loss:.4f}"

        if test_metric is not None:
            msg += f", test_metric: {test_metric:.4f}"

        msg += f", time: {epoch_time:.2f}s"

        if remaining_epochs > 0:
            msg += f", ETA: {estimated_remaining:.2f}s"

        print(msg)

    def run(self) -> None:
        """Run training for all epochs."""
        print(f"Starting training for {self.max_epoch} epochs...")
        start_time = time.time()

        self.step(epochs=self.max_epoch)

        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.2f} seconds")
        print(f"Average time per epoch: {total_time / self.max_epoch:.2f} seconds")

    def plot_history(
        self,
        history_types: str | list[str] = "loss",
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        colors: Optional[list[str]] = None,
        figsize: tuple[int, int] = (12, 6),
    ) -> None:
        """
        Plot training history over epochs.

        Args:
            history_types: Type(s) of history to plot. Can be a single string or list of strings.
                Options: "loss", "test_loss", "metric", "test_metric"
                Example: ["loss", "test_loss"] to plot both train and test loss
            ylabel: Label for y-axis (default: auto-generated based on history_types)
            title: Title of the plot (default: auto-generated based on history_types)
            colors: List of colors for each history type (default: None, uses matplotlib default)
            figsize: Figure size tuple (default: (12, 6))
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            )

        # Convert single string to list for uniform handling
        if isinstance(history_types, str):
            history_types = [history_types]

        # Select history based on type
        history_map = {
            "loss": self.train_loss_history,
            "test_loss": self.test_loss_history,
            "metric": self.train_metric_history,
            "test_metric": self.test_metric_history,
        }

        # Label map for automatic labeling
        label_map = {
            "loss": "Train",
            "test_loss": "Test",
            "metric": "Train",
            "test_metric": "Test",
        }

        # Validate all history types
        for history_type in history_types:
            if history_type not in history_map:
                raise ValueError(
                    f"Invalid history_type: {history_type}. "
                    f"Must be one of {list(history_map.keys())}"
                )

        # Check if any history has data
        histories_with_data = []
        for history_type in history_types:
            history = history_map[history_type]
            if len(history) > 0:
                histories_with_data.append(history_type)

        if len(histories_with_data) == 0:
            print(f"No data to plot for {history_types}.")
            return

        # Auto-generate ylabel and title if not provided
        if ylabel is None:
            # If all types are loss-related, use "Loss", if all are metric-related, use "Metric"
            if all("loss" in ht for ht in histories_with_data):
                ylabel = "Loss"
            elif all("metric" in ht for ht in histories_with_data):
                ylabel = "Metric"
            else:
                ylabel = "Value"

        if title is None:
            # Generate title based on what's being plotted
            if all("loss" in ht for ht in histories_with_data):
                title = (
                    "Training and Validation Loss"
                    if len(histories_with_data) > 1
                    else "Training Loss"
                )
            elif all("metric" in ht for ht in histories_with_data):
                title = (
                    "Training and Validation Metric"
                    if len(histories_with_data) > 1
                    else "Training Metric"
                )
            else:
                title = "Training History"

        # Default markers for different series
        default_markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

        # Plot
        plt.figure(figsize=figsize)

        for idx, history_type in enumerate(histories_with_data):
            history = history_map[history_type]
            epochs = range(1, len(history) + 1)

            plot_kwargs = {
                "linewidth": 2,
                "marker": default_markers[idx % len(default_markers)],
                "label": label_map[history_type],
            }

            if colors and idx < len(colors):
                plot_kwargs["color"] = colors[idx]

            plt.plot(epochs, history, **plot_kwargs)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
