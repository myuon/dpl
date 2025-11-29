import numpy as np
from dpl import Dataset


def get_spiral(num_samples=100, num_classes=3):
    data_size = num_samples * num_classes
    x = np.zeros((data_size, 2), dtype=np.float32)
    t = np.zeros(data_size, dtype=int)

    for j in range(num_classes):
        for i in range(num_samples):
            rate = i / num_samples
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_samples * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j

    # Shuffle
    indices = np.random.permutation(data_size)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral()
