from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from core.tensor import Tensor


def handle_broadcast(tensor: Tensor,
                     grad: np.ndarray):
    """
    Handles gradient summing when broadcasting np.ndarray
    Use with all binary operations that support broadcasting
    """
    dims_added = grad.ndim - tensor.data.ndim
    for _ in range(dims_added):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(tensor.shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
