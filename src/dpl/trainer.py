from typing import Callable, Optional
import time
from dpl.core import Variable, as_variable, no_grad, ndarray
from dpl.dataloaders import DataLoader
from dpl.layers import Layer
from dpl.optimizers import Optimizer


class Trainer:
    def __init__(
        self,
        model: Layer,
        optimizer: Optimizer,
        loss_fn: Callable[[Variable, Variable], Variable],
        metric_fn: Optional[Callable[[Variable, Variable], Variable]] = None,
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        max_epoch: int = 10,
        preprocess_fn: Optional[
            Callable[[ndarray, ndarray], tuple[ndarray, ndarray]]
        ] = None,
        on_epoch_start: Optional[Callable[["Trainer"], None]] = None,
        on_epoch_end: Optional[Callable[["Trainer"], None]] = None,
        on_batch_end: Optional[Callable[["Trainer"], None]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epoch = max_epoch
        self.preprocess_fn = preprocess_fn

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
            # Preprocess if needed
            if self.preprocess_fn is not None:
                x, t = self.preprocess_fn(x, t)

            # Convert to Variable
            x, t = as_variable(x), as_variable(t)

            # Forward
            y = self.model(x)
            loss = self.loss_fn(y, t)

            # Backward
            loss.backward()
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
                # Preprocess if needed
                if self.preprocess_fn is not None:
                    x, t = self.preprocess_fn(x, t)

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
