from __future__ import annotations
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Optional, Tuple
from abc import ABC, abstractmethod
from .autograd import *

PrimType = int | float | List
CPUType = PrimType | np.ndarray
GPUType = PrimType | cp.ndarray


# Returns true if t1 & t2 are same device backends, false otherwise
def same_device(t1, t2):
    return isinstance(t1, TensorCPUBackend) and isinstance(t2, TensorCPUBackend) \
        or (isinstance(t1, TensorGPUBackend) and isinstance(t2, TensorGPUBackend))


# Converts data to np.array for CPU use
def cpu_data_convert(data: Any):
    if not isinstance(data, CPUType):
        raise TypeError("CPU Tensor must be initialized from int, float, List, or np.ndarray")
    else:
        if isinstance(data, np.ndarray):
            return data
        return np.array(data, dtype=np.float64)


# Converts data to cp.array for GPU use
def gpu_data_convert(data: Any):
    if not isinstance(data, GPUType):
        raise TypeError("GPU Tensor must be initialized from int, float, List, or cp.ndarray")
    else:
        if isinstance(data, cp.ndarray):
            return data
        return cp.array(data, dtype=cp.float64)


# TODO: pls refactor this absolute mess & don't use inheritance
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
        self._cpu = isinstance(self, TensorCPUBackend)

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
    def backward(self, grad: Optional = None):
        raise NotImplementedError

    def unary_op(self, grad_type):
        grad_fn = grad_type(self.data)
        backend_cls = TensorCPUBackend if self._cpu else TensorGPUBackend
        return backend_cls(
            data=grad_fn.data,
            requires_grad=self.requires_grad,
            grad_fn=grad_fn,
            children=[self],
            leaf=False
        )

    def binary_op(self, other, grad_type):
        backend_cls = TensorCPUBackend if self._cpu else TensorGPUBackend
        if isinstance(other, PrimType):
            other = backend_cls(other)
        if same_device(self, other):
            grad_fn = grad_type(self.data, other.data)
            return backend_cls(data=grad_fn.data,
                               requires_grad=(self.requires_grad or other.requires_grad),
                               grad_fn=grad_fn,
                               children=[self, other],
                               leaf=False)
        else:
            raise RuntimeError("Device mismatch")

    def transpose(self):
        return self.unary_op(Transpose)

    def sum(self):
        return self.unary_op(Sum)

    def __neg__(self):
        return self.unary_op(Neg)

    def exp(self):
        return self.unary_op(Exp)

    def log(self):
        return self.unary_op(Log)

    def square(self):
        return self.unary_op(Square)

    def sqrt(self):
        return self.unary_op(Sqrt)

    def mean(self):
        return self.unary_op(Mean)

    def sin(self):
        return self.unary_op(Sin)

    def cos(self):
        return self.unary_op(Cos)

    def tan(self):
        return self.unary_op(Tan)

    def arcsin(self):
        return self.unary_op(Arcsin)

    def arccos(self):
        return self.unary_op(Arccos)

    def arctan(self):
        return self.unary_op(Arctan)

    def sinh(self):
        return self.unary_op(Sinh)

    def cosh(self):
        return self.unary_op(Cosh)

    def tanh(self):
        return self.unary_op(Tanh)

    def arcsinh(self):
        return self.unary_op(Arcsinh)

    def arccosh(self):
        return self.unary_op(Arccosh)

    def arctanh(self):
        return self.unary_op(Arctanh)

    def relu(self):
        return self.unary_op(ReLU)

    def __add__(self, other):
        return self.binary_op(other, Add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.binary_op(other, Sub)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        return self.binary_op(other, Mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.binary_op(other, Div)

    def __matmul__(self, other):
        return self.binary_op(other, MatMul)

    # TODO: Fix these
    def __iadd__(self, other):
        self.data += other.data
        return self

    def __isub__(self, other):
        self.data -= other.data
        return self

    def __imul__(self, other):
        self.data *= other.data
        return self

    # testing compound ops.
    def abs(self):
        return self.square().sqrt()

    def sigmoid(self):
        backend_cls = TensorCPUBackend if self._cpu else TensorGPUBackend
        return backend_cls(1) / ((-self).exp() + 1)


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
        if self.requires_grad:
            if grad is None:
                if self.data.ndim != 0:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                else:
                    grad = TensorCPUBackend(1)
            self.grad.data += grad.data
            if not self.leaf:
                gradients = self.grad_fn(self.grad.data)
                for child, gradient in zip(self.children, gradients):
                    if gradient is not None and child.requires_grad is True:
                        child.backward(gradient)
        else:
            raise RuntimeError(".backward() run on Tensor with requires_grad=False")


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
        if self.requires_grad:
            if grad is None:
                if cp.squeeze(self.data).ndim != 0:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                else:
                    grad = TensorGPUBackend(1)
            self.grad.data += grad.data
            if not self.leaf:
                gradients = self.grad_fn(self.grad.data)
                for child, gradient in zip(self.children, gradients):
                    if gradient is not None and child.requires_grad is True:
                        child.backward(gradient)
        else:
            raise RuntimeError(".backward() run on Tensor with requires_grad=False")
