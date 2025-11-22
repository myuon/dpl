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
        self.generation = 0

    def set_creator(self, func: "Function") -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        assert self.creator is not None

        stack = []
        visited = set()

        def add_visited(f: Function) -> None:
            if f not in visited:
                stack.append(f)
                visited.add(f)
                stack.sort(key=lambda func: func.generation)

        add_visited(self.creator)

        while stack:
            f = stack.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            assert len(gxs) == len(f.inputs)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_visited(x.creator)

    def cleargrad(self) -> None:
        self.grad = None
