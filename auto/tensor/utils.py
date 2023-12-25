import numpy as np


def handle_broadcast(tensor: np.ndarray,
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
