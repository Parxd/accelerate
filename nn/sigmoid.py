import numpy as np
from core.tensor import Tensor


class Sigmoid:
    def __init__(self):
        ...

    def __call__(self, data: Tensor) -> Tensor:
        return data.sigmoid()

    def __str__(self):
        return "Sigmoid()"
