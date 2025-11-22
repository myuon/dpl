import numpy as np
from utils import as_nparray
from variable import Variable


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_nparray(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
