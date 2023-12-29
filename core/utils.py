from typing import Tuple
import numpy as np
from core.tensor import Tensor


def zeros(shape: Tuple):
    return Tensor(np.zeros(shape))


def ones(shape: Tuple):
    return Tensor(np.ones(shape))


def zeros_like(tensor: Tensor):
    return zeros(tensor.shape)


def ones_like(tensor: Tensor):
    return ones(tensor.shape)
