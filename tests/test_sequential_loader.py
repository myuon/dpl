"""Tests for SequentialDataLoader class."""

import numpy as np
import pytest
import sys
from pathlib import Path

# src をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpl.datasets.dataloader import SequentialDataLoader


class DummySequentialDataset:
    """Simple dataset that returns sequential numbers for testing."""
    def __init__(self, size):
        self.data = np.arange(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]


class TestSequentialDataLoader:
    def test_init(self):
        """Test SequentialDataLoader initialization."""
        dataset = DummySequentialDataset(24)
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=5)

        assert loader.batch_size == 2
        assert loader.bptt_length == 5
        assert loader.data_size == 24
        assert loader.stream_length == 12  # 24 // 2
        assert loader.max_iter == 2  # (12 - 1) // 5

    def test_stream_division(self):
        """Test that data is correctly divided into streams."""
        dataset = DummySequentialDataset(24)
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=5)

        # First batch should contain:
        # stream0: [0, 1, 2, 3, 4]
        # stream1: [12, 13, 14, 15, 16]
        xs, ts = next(iter(loader))

        assert xs.shape == (2, 5)
        assert ts.shape == (2, 5)

        # Check stream0
        np.testing.assert_array_equal(xs[0], np.array([0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(ts[0], np.array([1, 2, 3, 4, 5]))

        # Check stream1
        np.testing.assert_array_equal(xs[1], np.array([12, 13, 14, 15, 16]))
        np.testing.assert_array_equal(ts[1], np.array([13, 14, 15, 16, 17]))

    def test_temporal_continuity(self):
        """Test that temporal continuity is maintained within streams."""
        dataset = DummySequentialDataset(24)
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=5)

        batches = list(loader)

        # Second batch should continue from where the first batch ended
        # stream0: [5, 6, 7, 8, 9]
        # stream1: [17, 18, 19, 20, 21]
        xs_batch2, ts_batch2 = batches[1]

        np.testing.assert_array_equal(xs_batch2[0], np.array([5, 6, 7, 8, 9]))
        np.testing.assert_array_equal(ts_batch2[0], np.array([6, 7, 8, 9, 10]))

        np.testing.assert_array_equal(xs_batch2[1], np.array([17, 18, 19, 20, 21]))
        np.testing.assert_array_equal(ts_batch2[1], np.array([18, 19, 20, 21, 22]))

    def test_target_is_one_step_ahead(self):
        """Test that targets are one time step ahead of inputs."""
        dataset = DummySequentialDataset(20)
        loader = SequentialDataLoader(dataset, batch_size=1, bptt_length=3)

        xs, ts = next(iter(loader))

        # For each time step, target should be input + 1
        for i in range(xs.shape[1]):
            assert ts[0, i] == xs[0, i] + 1

    def test_iteration_count(self):
        """Test that the number of iterations matches max_iter."""
        dataset = DummySequentialDataset(24)
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=5)

        batches = list(loader)

        assert len(batches) == loader.max_iter
        assert len(batches) == 2

    def test_reset_and_reiteration(self):
        """Test that loader can be reset and iterated again."""
        dataset = DummySequentialDataset(24)
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=5)

        # First iteration
        first_batches = list(loader)

        # Second iteration (should reset automatically)
        second_batches = list(loader)

        assert len(first_batches) == len(second_batches)

        # First batches should be identical
        for (xs1, ts1), (xs2, ts2) in zip(first_batches, second_batches):
            np.testing.assert_array_equal(xs1, xs2)
            np.testing.assert_array_equal(ts1, ts2)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        dataset = DummySequentialDataset(30)

        # batch_size=3 -> 3 streams of length 10 each
        loader = SequentialDataLoader(dataset, batch_size=3, bptt_length=4)

        assert loader.stream_length == 10
        assert loader.max_iter == 2  # (10 - 1) // 4

        xs, ts = next(iter(loader))
        assert xs.shape == (3, 4)

        # Check all three streams
        np.testing.assert_array_equal(xs[0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(xs[1], np.array([10, 11, 12, 13]))
        np.testing.assert_array_equal(xs[2], np.array([20, 21, 22, 23]))

    def test_different_bptt_lengths(self):
        """Test with different BPTT lengths."""
        dataset = DummySequentialDataset(20)

        # bptt_length=10
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=10)

        assert loader.max_iter == 0  # (10 - 1) // 10 = 0

        # bptt_length=3
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=3)

        assert loader.max_iter == 3  # (10 - 1) // 3 = 3

        batches = list(loader)
        assert len(batches) == 3

    def test_next_method(self):
        """Test the next() method."""
        dataset = DummySequentialDataset(24)
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=5)

        xs1, ts1 = loader.next()
        xs2, ts2 = loader.next()

        # Should return different batches
        assert not np.array_equal(xs1, xs2)

        # After max_iter, should raise StopIteration
        with pytest.raises(StopIteration):
            loader.next()

    def test_single_batch_size(self):
        """Test with batch_size=1."""
        dataset = DummySequentialDataset(15)
        loader = SequentialDataLoader(dataset, batch_size=1, bptt_length=5)

        xs, ts = next(iter(loader))

        assert xs.shape == (1, 5)
        np.testing.assert_array_equal(xs[0], np.array([0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(ts[0], np.array([1, 2, 3, 4, 5]))

    def test_gpu_flag(self):
        """Test GPU flag initialization."""
        dataset = DummySequentialDataset(20)

        loader_cpu = SequentialDataLoader(dataset, batch_size=2, bptt_length=5, gpu=False)
        assert loader_cpu.gpu is False

        loader_gpu = SequentialDataLoader(dataset, batch_size=2, bptt_length=5, gpu=True)
        assert loader_gpu.gpu is True

    def test_to_cpu_to_gpu(self):
        """Test to_cpu() and to_gpu() methods."""
        dataset = DummySequentialDataset(20)
        loader = SequentialDataLoader(dataset, batch_size=2, bptt_length=5, gpu=True)

        assert loader.gpu is True

        loader.to_cpu()
        assert loader.gpu is False

        loader.to_gpu()
        assert loader.gpu is True
