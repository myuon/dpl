from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from function import Function


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.grad: np.ndarray | None = None
        self.creator: "Function | None" = None

    def set_creator(self, func: "Function") -> None:
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        if self.creator is None:
            return

        stack = [self.creator]
        while stack:
            f = stack.pop()
            x, y = f.input, f.output
            assert y.grad is not None
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                stack.append(x.creator)
