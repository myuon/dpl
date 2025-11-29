import numpy as np
from dpl.datasets.dataset import Dataset
from dpl.core import metal


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = int(np.ceil(self.data_size / batch_size))
        self.gpu = gpu
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.indices = np.random.permutation(self.data_size)
        else:
            self.indices = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        xp = metal.jnp if self.gpu else np

        i, batch_size = self.iteration, self.batch_size
        batch_indices = self.indices[i * batch_size : (i + 1) * batch_size]
        batch = [self.dataset[idx] for idx in batch_indices]
        x = xp.array([item[0] for item in batch])
        y = xp.array([item[1] for item in batch])
        self.iteration += 1
        return x, y

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True
