from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from core.tensor import Tensor


class NegBackward:
    def __init__(self,
                 child: Tensor):
        self.child = child
        self.data = neg(child)

    def __call__(self,
                 grad: np.ndarray):
        return (-grad,)


def neg(tensor: Tensor):
    from core.tensor import Tensor
    return Tensor(np.array(-1)) * tensor
