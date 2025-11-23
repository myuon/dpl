"""Tests for Dataset class."""

import numpy as np
import pytest
import sys
from pathlib import Path

# src をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpl.datasets import Dataset


class TestDataset:
    def test_init(self):
        """Test Dataset initialization"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 1, 0])

        dataset = Dataset(data, labels)

        assert dataset.data.shape == (3, 2)
        assert dataset.label.shape == (3,)
        assert len(dataset) == 3

    def test_init_length_mismatch(self):
        """Test that Dataset raises ValueError when data and label lengths don't match"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 1])  # Different length

        with pytest.raises(ValueError, match="Data and label must have the same length"):
            Dataset(data, labels)

    def test_getitem_single(self):
        """Test getting a single item"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 1, 0])
        dataset = Dataset(data, labels)

        x, y = dataset[0]
        np.testing.assert_array_equal(x, np.array([1, 2]))
        assert y == 0

        x, y = dataset[1]
        np.testing.assert_array_equal(x, np.array([3, 4]))
        assert y == 1

        x, y = dataset[2]
        np.testing.assert_array_equal(x, np.array([5, 6]))
        assert y == 0

    def test_getitem_slice(self):
        """Test getting items with slice"""
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        labels = np.array([0, 1, 0, 1])
        dataset = Dataset(data, labels)

        x_batch, y_batch = dataset[0:2]
        np.testing.assert_array_equal(x_batch, np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(y_batch, np.array([0, 1]))

        x_batch, y_batch = dataset[1:3]
        np.testing.assert_array_equal(x_batch, np.array([[3, 4], [5, 6]]))
        np.testing.assert_array_equal(y_batch, np.array([1, 0]))

    def test_getitem_array_indices(self):
        """Test getting items with array of indices"""
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        labels = np.array([0, 1, 0, 1])
        dataset = Dataset(data, labels)

        indices = np.array([0, 2, 3])
        x_batch, y_batch = dataset[indices]

        np.testing.assert_array_equal(x_batch, np.array([[1, 2], [5, 6], [7, 8]]))
        np.testing.assert_array_equal(y_batch, np.array([0, 0, 1]))

    def test_getitem_negative_index(self):
        """Test getting items with negative index"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 1, 0])
        dataset = Dataset(data, labels)

        x, y = dataset[-1]
        np.testing.assert_array_equal(x, np.array([5, 6]))
        assert y == 0

        x, y = dataset[-2]
        np.testing.assert_array_equal(x, np.array([3, 4]))
        assert y == 1

    def test_len(self):
        """Test __len__ method"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 1, 0])
        dataset = Dataset(data, labels)

        assert len(dataset) == 3

        data2 = np.array([[1], [2], [3], [4], [5]])
        labels2 = np.array([0, 1, 0, 1, 0])
        dataset2 = Dataset(data2, labels2)

        assert len(dataset2) == 5

    def test_with_different_shapes(self):
        """Test Dataset with different data shapes"""
        # 1D features
        data = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(data, labels)

        x, y = dataset[0]
        assert x == 1
        assert y == 0

        # 3D data (e.g., images)
        data = np.random.randn(10, 28, 28)
        labels = np.random.randint(0, 10, 10)
        dataset = Dataset(data, labels)

        x, y = dataset[0]
        assert x.shape == (28, 28)
        assert isinstance(y, (int, np.integer))
