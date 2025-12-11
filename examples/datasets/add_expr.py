import numpy as np
from dpl import Dataset


class AddExpr(Dataset):
    """
    Addition Expression dataset for seq2seq learning.

    Generates samples of the form:
    - Input: "123+456" (addition expression as character sequence)
    - Target: "579" (result as character sequence)

    Character vocabulary:
        0-9: digits
        '+': plus sign
        ' ': padding
        '_': start token (for decoder)
    """

    # Character to ID mapping
    CHARS = "0123456789+ _"
    CHAR2ID = {c: i for i, c in enumerate(CHARS)}
    ID2CHAR = {i: c for i, c in enumerate(CHARS)}
    PAD_ID = CHAR2ID[" "]
    START_ID = CHAR2ID["_"]
    VOCAB_SIZE = len(CHARS)

    def __init__(
        self,
        num_samples: int = 50000,
        max_digits: int = 3,
        train: bool = True,
        seed: int | None = None,
        reverse_input: bool = False,
    ):
        """
        Args:
            num_samples: Number of samples to generate
            max_digits: Maximum number of digits for each operand
            train: Whether this is training set
            seed: Random seed for reproducibility
            reverse_input: If True, reverse the input sequence (improves seq2seq learning)
        """
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.seed = seed
        self.reverse_input = reverse_input
        # Input length: max_digits + 1 ('+') + max_digits = 2 * max_digits + 1
        self.input_len = 2 * max_digits + 1
        # Output length: max_digits + 1 (for possible carry)
        self.output_len = max_digits + 1
        super().__init__(train=train)

    def prepare(self):
        if self.seed is not None:
            np.random.seed(self.seed if self.train else self.seed + 1)

        self.data = []
        self.label = []

        for _ in range(self.num_samples):
            # Generate random numbers
            max_val = 10**self.max_digits - 1
            a = np.random.randint(0, max_val + 1)
            b = np.random.randint(0, max_val + 1)
            result = a + b

            # Create input string (padded to fixed length)
            # Format: right-aligned with padding
            input_str = f"{a}+{b}"
            input_str = input_str.rjust(self.input_len)

            # Create output string (padded to fixed length)
            output_str = str(result)
            output_str = output_str.rjust(self.output_len)

            # Convert to IDs
            input_ids = np.array(
                [self.CHAR2ID[c] for c in input_str], dtype=np.int32
            )
            output_ids = np.array(
                [self.CHAR2ID[c] for c in output_str], dtype=np.int32
            )

            # Reverse input if requested
            if self.reverse_input:
                input_ids = input_ids[::-1].copy()

            self.data.append(input_ids)
            self.label.append(output_ids)

    @classmethod
    def decode(cls, ids: np.ndarray) -> str:
        """Convert ID sequence back to string."""
        return "".join(cls.ID2CHAR[int(i)] for i in ids)

    @classmethod
    def encode(cls, s: str) -> np.ndarray:
        """Convert string to ID sequence."""
        return np.array([cls.CHAR2ID[c] for c in s], dtype=np.int32)
