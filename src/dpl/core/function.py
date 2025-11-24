from dpl.core.utils import as_nparray, ndarray
import weakref
from dpl.core.config import Config
from dpl.core.variable import as_variable, Variable


class Function:
    def __call__(self, *_inputs: ndarray | Variable):
        inputs = [as_variable(x) for x in _inputs]

        xs = [x.data_required for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [as_variable(as_nparray(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs: list[weakref.ReferenceType[Variable]] = [
                weakref.ref(output) for output in outputs
            ]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: ndarray) -> ndarray:
        raise NotImplementedError()

    def backward(self, *gys: Variable) -> Variable | tuple[Variable, ...]:
        raise NotImplementedError()


class UnaryFunction(Function):
    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result


class BinaryFunction(Function):
    def apply(self, x0: Variable, x1: Variable) -> Variable:
        result = super().__call__(x0, x1)
        assert isinstance(result, Variable)
        return result
