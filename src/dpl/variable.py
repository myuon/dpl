from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from utils import unwrap

if TYPE_CHECKING:
    from function import Function


class Variable:
    def __init__(self, data: np.ndarray, name: str | None = None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.name = name
        self.grad: np.ndarray | None = None
        self.creator: "Function | None" = None
        self.generation = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def set_creator(self, func: "Function") -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        assert self.creator is not None

        stack: list[Function] = []
        visited: set[Function] = set()

        def add_visited(f: Function) -> None:
            if f not in visited:
                stack.append(f)
                visited.add(f)
                stack.sort(key=lambda func: func.generation)

        add_visited(self.creator)

        while stack:
            f = stack.pop()
            gys = [unwrap(unwrap(output()).grad) for output in f.outputs]
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

            if not retain_grad:
                for y in f.outputs:
                    unwrap(y()).grad = None

    def cleargrad(self) -> None:
        self.grad = None

    def __mul__(self, other: Variable) -> Variable:
        raise NotImplementedError()

    def __add__(self, other: Variable) -> Variable:
        raise NotImplementedError()
