import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Optional, Tuple
from abc import ABC, abstractmethod

PrimType = int | float | List
CPUType = PrimType | np.ndarray
GPUType = PrimType | cp.ndarray

# def convert_to_operable(other: Any) -> Tensor:
#     if isinstance(other, Tensor):
#         return other
#     if isinstance(other, ScalarType):
#         return Tensor(np.array(other))
#     elif isinstance(other, np.ndarray):
#         return Tensor(other)
#     raise TypeError(f"unsupported operand type(s) for Tensor and {type(other)}")


def cpu_data_convert(data: Any):
    if not isinstance(data, CPUType):
        raise TypeError("CPU Tensor must be initialized from int, float, List, or np.ndarray")
    else:
        if isinstance(data, np.ndarray):
            return data
        return np.array(data, dtype=np.float64)


def gpu_data_convert(data: Any):
    if not isinstance(data, GPUType):
        raise TypeError("GPU Tensor must be initialized from int, float, List, or cp.ndarray")
    else:
        if isinstance(data, cp.ndarray):
            return data
        return cp.array(data, dtype=np.float64)


class TensorBackend(ABC):
    def __init__(self,
                 data: np.ndarray | cp.ndarray,
                 grad=None,
                 requires_grad: bool = False,
                 grad_fn: Callable = None,
                 children: List = None,
                 leaf: bool = True) -> None:
        if children is None:
            children = []
        self.data = data
        self.grad = grad
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.children = children
        self.leaf = leaf

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.data, **kwargs)

    def __eq__(self, other):
        return np.array_equal(self.data, other.data) and self.data.dtype == other.data.dtype

    def __str__(self):
        return (f"Tensor("
                f"{np.array2string(self.data, prefix='Tensor(')}, requires_grad={self.requires_grad})")

    def __repr__(self):
        return f"{self.data}"

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def random(cls, shape: Tuple, requires_grad: bool):
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad):
        raise NotImplementedError


class TensorCPUBackend(TensorBackend):
    def __init__(self,
                 data: CPUType,
                 requires_grad: bool = False,
                 grad_fn: Callable = None,
                 children: List = None,
                 leaf: bool = True) -> None:
        super().__init__(
            cpu_data_convert(data),
            TensorCPUBackend(np.zeros_like(data, dtype=np.float64)) if requires_grad else None,
            requires_grad,
            grad_fn,
            children,
            leaf
        )

    @classmethod
    def random(cls, shape: Tuple, requires_grad: bool = False):
        return cls(np.random.randn(*shape), requires_grad=requires_grad)

    def zero_grad(self):
        self.grad.data = np.zeros_like(self.data, dtype=np.float64)

    def backward(self, grad: Optional = None):
        ...


class TensorGPUBackend(TensorBackend):
    def __init__(self,
                 data: GPUType,
                 requires_grad: bool = False,
                 grad_fn: Callable = None,
                 children: List = None,
                 leaf: bool = True) -> None:
        super().__init__(
            gpu_data_convert(data),
            TensorGPUBackend(cp.zeros_like(data, dtype=np.float64)) if requires_grad else None,
            requires_grad,
            grad_fn,
            children,
            leaf
        )

    @classmethod
    def random(cls, shape: Tuple, requires_grad: bool = False):
        return cls(cp.random.randn(*shape), requires_grad=requires_grad)

    def zero_grad(self):
        self.grad.data = cp.zeros_like(self.data, dtype=cp.float64)

    def backward(self, grad: Optional = None):
        ...
