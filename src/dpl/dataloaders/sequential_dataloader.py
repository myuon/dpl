import numpy as np
from dpl.datasets.dataset import Dataset
from dpl.core import metal


class SequentialDataLoader:
    """
    DataLoader for sequential data with BPTT support.

    Splits the dataset into batch_size streams and reads them in parallel,
    maintaining temporal continuity within each stream.
    """
    def __init__(self, dataset: Dataset, batch_size: int, bptt_length: int, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bptt_length = bptt_length
        self.data_size = len(dataset)
        self.gpu = gpu

        # Calculate the length of each stream
        self.stream_length = self.data_size // batch_size
        # Calculate max iterations (number of bptt chunks per stream)
        self.max_iter = (self.stream_length - 1) // bptt_length

        self.reset()

    def reset(self):
        self.iteration = 0
        # Starting position for each stream
        self.stream_positions = [i * self.stream_length for i in range(self.batch_size)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        xp = metal.jnp if self.gpu else np

        # Collect sequences from each stream
        xs_batch = []
        ts_batch = []

        for stream_idx in range(self.batch_size):
            # Calculate the starting position in the original dataset
            start_pos = stream_idx * self.stream_length + self.iteration * self.bptt_length

            # Collect bptt_length steps
            xs_seq = []
            ts_seq = []
            for step in range(self.bptt_length):
                pos = start_pos + step
                if pos + 1 < (stream_idx + 1) * self.stream_length:
                    x, _ = self.dataset[pos]
                    _, t = self.dataset[pos + 1]
                    xs_seq.append(x)
                    ts_seq.append(t)
                else:
                    # If we run out of data, stop early
                    break

            if xs_seq:  # Only add if we have data
                xs_batch.append(xs_seq)
                ts_batch.append(ts_seq)

        self.iteration += 1

        # Convert to arrays: shape (batch_size, bptt_length, ...)
        xs = xp.array(xs_batch)
        ts = xp.array(ts_batch)

        return xs, ts

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True
