import numpy as np
from typing import Any, Callable


class Dataset:
    def __init__(self, train=True, transform: Callable | None = None):
        self.train = train
        self.transform = transform if transform is not None else lambda x: x
        self.data = {}
        self.label = None

        self.prepare()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None

        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass
