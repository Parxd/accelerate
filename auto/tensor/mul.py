from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .utils import handle_broadcast
if TYPE_CHECKING:
    from core.tensor import Tensor


class MulBackward:
    def __init__(self,
                 left_child: Tensor,
                 right_child: Tensor):
        self.left_child = left_child
        self.right_child = right_child
        self.data = mul(left_child, right_child)

    def __call__(self,
                 grad: np.ndarray):
        l_grad = handle_broadcast(self.left_child, (grad * self.right_child.data))
        r_grad = handle_broadcast(self.right_child, (grad * self.left_child.data))
        return l_grad, r_grad


def mul(left_child: Tensor,
        right_child: Tensor):
    return left_child.data * right_child.data
