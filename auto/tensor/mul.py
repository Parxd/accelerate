import numpy as np
from .utils import handle_broadcast


class MulBackward:
    def __init__(self,
                 left_child: np.ndarray,
                 right_child: np.ndarray):
        self.left_child = left_child
        self.right_child = right_child
        self.data = mul(left_child, right_child)

    def __call__(self,
                 grad: np.ndarray):
        l_grad = handle_broadcast(self.left_child, (grad * self.right_child))
        r_grad = handle_broadcast(self.right_child, (grad * self.left_child))
        return l_grad, r_grad


def mul(left_child: np.ndarray,
        right_child: np.ndarray):
    return left_child * right_child
