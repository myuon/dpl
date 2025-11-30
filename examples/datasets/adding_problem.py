import numpy as np
from dpl import Dataset


class AddingProblem(Dataset):
    """
    Adding Problem dataset for testing RNN long-term dependencies.

    Each sample consists of:
    - Input: T x 2 array where
        - First column: random values in [0, 1]
        - Second column: mask with exactly 2 positions set to 1
    - Target: sum of the two values where mask is 1
    """

    def __init__(self, num_samples=10000, sequence_length=50, train=True):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        super().__init__(train=train)

    def prepare(self):
        self.data = []
        self.label = []

        for _ in range(self.num_samples):
            # Generate random values in [0, 1]
            values = np.random.uniform(0, 1, size=self.sequence_length)

            # Create mask with exactly 2 positions set to 1
            mask = np.zeros(self.sequence_length)
            positions = np.random.choice(self.sequence_length, size=2, replace=False)
            mask[positions] = 1

            # Stack values and mask as 2D input
            x = np.stack([values, mask], axis=1)  # (T, 2)

            # Target is the sum of values where mask is 1
            target = np.sum(values[positions])

            self.data.append(x.astype(np.float64))
            self.label.append(np.array([target], dtype=np.float64))
