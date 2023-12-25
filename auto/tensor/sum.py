import numpy as np


class SumBackward:
    def __init__(self,
                 child: np.ndarray):
        self.child = child
        self.data = tensor_sum(child)

    def __call__(self,
                 grad: np.ndarray):
        return grad


def tensor_sum(tensor: np.ndarray):
    return tensor.sum()
