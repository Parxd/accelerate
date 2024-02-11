from abc import ABC, abstractmethod
import numpy as np
import cupy as cp


class Node(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradient: np.ndarray | cp.ndarray):
        raise NotImplementedError
