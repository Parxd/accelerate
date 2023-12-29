from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from core.tensor import Tensor


class SumBackward:
    def __init__(self,
                 child: Tensor):
        self.child = child
        self.data = tensor_sum(child)

    def __call__(self,
                 grad: np.ndarray):
        return (grad,)


def tensor_sum(tensor: Tensor):
    return tensor.data.sum()
