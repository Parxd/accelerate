import numpy as np
from .utils import handle_broadcast


class AddBackward:
    def __init__(self,
                 left_child: np.ndarray,
                 right_child: np.ndarray):
        self.left_child = left_child
        self.right_child = right_child
        self.data = add(left_child, right_child)

    def __call__(self,
                 grad: np.ndarray):
        l_grad = handle_broadcast(self.left_child, grad)
        r_grad = handle_broadcast(self.right_child, grad)
        return l_grad, r_grad


def add(tensor1: np.ndarray,
        tensor2: np.ndarray):
    return tensor1 + tensor2
