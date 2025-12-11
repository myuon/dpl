import numpy as np
from dpl import Dataset


class DateFormat(Dataset):
    """
    Date Format conversion dataset for seq2seq learning.

    Generates samples converting various date formats to YYYY-MM-DD:
    - Input: Various formats like "January 5, 2021", "1/5/21", "2021/01/05", etc.
    - Target: "YYYY-MM-DD" format (e.g., "2021-01-05")

    Input formats:
        1. "january 5, 2021" - full month name
        2. "jan 5, 2021" - abbreviated month name
        3. "1/5/2021" - M/D/YYYY (US format)
        4. "01/05/2021" - MM/DD/YYYY
        5. "2021/01/05" - YYYY/MM/DD (ISO-like)
        6. "5 january 2021" - D month YYYY
        7. "5-jan-2021" - D-mon-YYYY
    """

    MONTHS_FULL = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    MONTHS_ABBR = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    # Character vocabulary: digits, lowercase letters, separators, special tokens
    CHARS = "0123456789abcdefghijklmnopqrstuvwxyz/-, _"
    CHAR2ID = {c: i for i, c in enumerate(CHARS)}
    ID2CHAR = {i: c for i, c in enumerate(CHARS)}
    PAD_ID = CHAR2ID[" "]
    START_ID = CHAR2ID["_"]
    VOCAB_SIZE = len(CHARS)

    # Fixed lengths
    INPUT_LEN = 20  # Max input like "september 30, 2021"
    OUTPUT_LEN = 10  # "YYYY-MM-DD"

    def __init__(
        self,
        num_samples: int = 50000,
        train: bool = True,
        seed: int | None = None,
        year_range: tuple[int, int] = (1900, 2100),
        reverse_input: bool = False,
    ):
        """
        Args:
            num_samples: Number of samples to generate
            train: Whether this is training set
            seed: Random seed for reproducibility
            year_range: Range of years to generate (inclusive)
            reverse_input: If True, reverse the input sequence
        """
        self.num_samples = num_samples
        self.seed = seed
        self.year_range = year_range
        self.reverse_input = reverse_input
        self.input_len = self.INPUT_LEN
        self.output_len = self.OUTPUT_LEN
        super().__init__(train=train)

    def _random_date(self, rng: np.random.Generator) -> tuple[int, int, int]:
        """Generate a random valid date."""
        year = rng.integers(self.year_range[0], self.year_range[1] + 1)
        month = rng.integers(1, 13)

        # Days per month (simplified - not handling leap years precisely)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            max_day = 29
        else:
            max_day = days_in_month[month - 1]

        day = rng.integers(1, max_day + 1)
        return year, month, day

    def _format_input(self, year: int, month: int, day: int, fmt: int) -> str:
        """Format date in various input styles."""
        month_full = self.MONTHS_FULL[month - 1]
        month_abbr = self.MONTHS_ABBR[month - 1]

        if fmt == 0:
            # "january 5, 2021"
            return f"{month_full} {day}, {year}"
        elif fmt == 1:
            # "jan 5, 2021"
            return f"{month_abbr} {day}, {year}"
        elif fmt == 2:
            # "1/5/2021" (M/D/YYYY)
            return f"{month}/{day}/{year}"
        elif fmt == 3:
            # "01/05/2021" (MM/DD/YYYY)
            return f"{month:02d}/{day:02d}/{year}"
        elif fmt == 4:
            # "2021/01/05" (YYYY/MM/DD)
            return f"{year}/{month:02d}/{day:02d}"
        elif fmt == 5:
            # "5 january 2021"
            return f"{day} {month_full} {year}"
        elif fmt == 6:
            # "5-jan-2021"
            return f"{day}-{month_abbr}-{year}"
        else:
            raise ValueError(f"Unknown format: {fmt}")

    def _format_output(self, year: int, month: int, day: int) -> str:
        """Format date as YYYY-MM-DD."""
        return f"{year}-{month:02d}-{day:02d}"

    def _encode(self, s: str, length: int) -> np.ndarray:
        """Encode string to ID sequence with padding."""
        s = s.lower()
        s = s.ljust(length)[:length]  # Pad or truncate
        return np.array([self.CHAR2ID[c] for c in s], dtype=np.int32)

    def prepare(self):
        seed = self.seed if self.seed is not None else None
        if seed is not None and not self.train:
            seed = seed + 1
        rng = np.random.default_rng(seed)

        self.data = []
        self.label = []

        num_formats = 7
        for i in range(self.num_samples):
            year, month, day = self._random_date(rng)
            fmt = i % num_formats  # Cycle through formats

            input_str = self._format_input(year, month, day, fmt)
            output_str = self._format_output(year, month, day)

            input_ids = self._encode(input_str, self.input_len)
            output_ids = self._encode(output_str, self.output_len)

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
        s = s.lower()
        return np.array([cls.CHAR2ID[c] for c in s], dtype=np.int32)
