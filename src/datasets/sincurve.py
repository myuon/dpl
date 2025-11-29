import numpy as np
from dpl import Dataset


class SinCurve(Dataset):
    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]
