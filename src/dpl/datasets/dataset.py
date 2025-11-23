import numpy as np
from typing import Any


class Dataset:
    def __init__(self, train=True):
        self.train = train
        self.data = {}
        self.label = None

        self.prepare()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        assert np.isscalar(index)
        if self.label is None:
            return self.data[index], None

        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass
