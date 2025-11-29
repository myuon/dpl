from dpl.core import Variable, Function, ndarray, get_array_module


class Concat(Function):
    """Concatenate variables along a specified axis."""

    def __init__(self, axis: int = 0):
        self.axis = axis

    def forward(self, *xs: ndarray) -> ndarray:
        xp = get_array_module(xs[0])
        return xp.concatenate(xs, axis=self.axis)

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        (gy,) = gys
        xp = get_array_module(gy.data)

        # Split the gradient back to match the input shapes
        shapes = [x.shape[self.axis] for x in self.inputs]
        gxs = xp.split(gy.data, xp.cumsum(xp.array(shapes[:-1])), axis=self.axis)

        return tuple(Variable(gx) for gx in gxs)


def concat(variables: list[Variable | ndarray], axis: int = 0) -> Variable:
    """
    Concatenate variables along a specified axis.

    Args:
        variables: List of variables to concatenate
        axis: Axis along which to concatenate

    Returns:
        Concatenated variable
    """
    from dpl.core import as_variable

    variables = [as_variable(v) for v in variables]
    return Concat(axis=axis)(*variables)
