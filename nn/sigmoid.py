from core.tensor import Tensor
from .layer import Layer


class Sigmoid(Layer):
    def __init__(self):
        ...

    def __call__(self, data: Tensor) -> Tensor:
        return data.sigmoid()

    def __str__(self):
        return "Sigmoid()"

    def zero_grad(self):
        ...
