import numpy as np


class SumBackward:
    def __init__(self,
                 tensor: np.ndarray,
                 grad):
        self.child = tensor
        self.data = tensor_sum(tensor)
        self.grad = grad

    @staticmethod
    def compute_grad(grad):
        return grad


def tensor_sum(tensor):
    return tensor.sum()
