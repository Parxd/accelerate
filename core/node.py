from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import cupy as cp

Array = np.ndarray | cp.ndarray


class Node(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Array:
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradient: Array) -> Array:
        raise NotImplementedError
