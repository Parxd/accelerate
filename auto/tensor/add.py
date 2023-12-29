from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .utils import handle_broadcast
if TYPE_CHECKING:
    from core.tensor import Tensor


class AddBackward:
    def __init__(self,
                 left_child: Tensor,
                 right_child: Tensor):
        self.left_child = left_child
        self.right_child = right_child
        self.data = add(left_child, right_child)

    def __call__(self,
                 grad: np.ndarray):
        l_grad = handle_broadcast(self.left_child, grad)
        r_grad = handle_broadcast(self.right_child, grad)
        return l_grad, r_grad


def add(tensor1: Tensor,
        tensor2: Tensor):
    return tensor1.data + tensor2.data
