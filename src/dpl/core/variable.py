from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from dpl.core.utils import unwrap, ndarray, ndarray_types
from dpl.core.config import use_config


if TYPE_CHECKING:
    from dpl.core.function import Function


class Variable:
    __array_priority__ = 200

    def __init__(self, data: ndarray | None, name: str | None = None) -> None:
        if data is not None:
            if not isinstance(data, ndarray_types):
                raise TypeError(f"{type(data)} is not supported.")

        self.data: ndarray | None = data
        self.name = name
        self.grad: Variable | None = None
        self.creator: "Function | None" = None
        self.generation = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data_required.shape

    @property
    def ndim(self) -> int:
        return self.data_required.ndim

    @property
    def size(self) -> int:
        return self.data_required.size

    @property
    def dtype(self) -> np.dtype:
        return self.data_required.dtype

    @property
    def data_required(self) -> ndarray:
        assert (
            self.data is not None
        ), "data is None. Please set data before using data_required."
        return self.data

    @property
    def grad_required(self) -> Variable:
        assert (
            self.grad is not None
        ), "grad is None. Please set grad before using grad_required."
        return self.grad

    def __len__(self) -> int:
        return len(self.data_required)

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def set_creator(self, func: "Function") -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False) -> None:
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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

            with use_config("enable_backprop", create_graph):
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

    def to_cpu(self):
        from dpl.core import metal

        if self.data is not None:
            self.data = metal.as_numpy(self.data)

    def to_gpu(self):
        from dpl.core import metal

        if self.data is not None:
            self.data = metal.as_jax(self.data)

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is not None:
            stack = [self.creator]
            while stack:
                f = stack.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        stack.append(x.creator)
                        x.unchain()

    def __mul__(self, other: Variable | ndarray | int | float | np.number) -> Variable:
        raise NotImplementedError()

    def __add__(self, other: Variable | ndarray | int | float | np.number) -> Variable:
        raise NotImplementedError()

    def __rmul__(self, other: Variable | ndarray | int | float | np.number) -> Variable:
        raise NotImplementedError()

    def __radd__(self, other: Variable | ndarray | int | float | np.number) -> Variable:
        raise NotImplementedError()

    def __neg__(self) -> Variable:
        raise NotImplementedError()

    def __sub__(self, other: Variable | ndarray | int | float | np.number) -> Variable:
        raise NotImplementedError()

    def __rsub__(self, other: Variable | ndarray | int | float | np.number) -> Variable:
        raise NotImplementedError()

    def __truediv__(
        self, other: Variable | ndarray | int | float | np.number
    ) -> Variable:
        raise NotImplementedError()

    def __rtruediv__(
        self, other: Variable | ndarray | int | float | np.number
    ) -> Variable:
        raise NotImplementedError()

    def __pow__(self, exponent: float) -> Variable:
        raise NotImplementedError()

    def reshape(self, *shape: int) -> Variable:
        from dpl.functions import reshape

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def transpose(self, *axes: int) -> Variable:
        from dpl.functions import transpose

        return transpose(self, *axes)

    @property
    def T(self) -> Variable:
        return self.transpose()

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Variable:
        from dpl.functions import sum

        return sum(self, axis, keepdims)

    def max(self, axis: int | None = None, keepdims: bool = False) -> Variable:
        from dpl.functions import max

        return max(self, axis, keepdims)

    def __getitem__(self: Variable, slices: tuple | int) -> Variable:
        from dpl.functions import get_item

        return get_item(self, slices)


def as_variable(obj: ndarray | Variable) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
