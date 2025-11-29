import numpy as np
from dpl import Dataset


class MNIST(Dataset):
    def prepare(self):
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1)
        data = mnist.data.values.astype(np.float32) / 255.0
        label = mnist.target.values.astype(int)

        if self.train:
            self.data = data[:60000]
            self.label = label[:60000]
        else:
            self.data = data[60000:]
            self.label = label[60000:]
