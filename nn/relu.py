from core.tensor import Tensor
from .layer import Layer


class ReLU(Layer):
    def __init__(self):
        ...

    def __call__(self, data: Tensor) -> Tensor:
        return data.relu()

    def __str__(self):
        return "ReLU()"

    def zero_grad(self):
        ...
