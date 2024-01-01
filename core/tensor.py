from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from .autograd import *

ScalarType = int | float
PrimType = ScalarType | List
TensorType = PrimType | np.ndarray


def convert_to_array(data: Any):
    if not isinstance(data, TensorType):
        raise TypeError("Tensor must be initialized from int, float, List, or np.ndarray")
    if isinstance(data, PrimType):
        return np.array(data, dtype=np.float64)
    return data


def convert_to_operable(other: Any):
    if isinstance(other, Tensor):
        return other
    if isinstance(other, ScalarType):
        return Tensor(np.array(other))
    elif isinstance(other, np.ndarray):
        return Tensor(other)
    raise TypeError(f"unsupported operand type(s) for Tensor and {type(other)}")


class Tensor:
    def __init__(self,
                 data: TensorType,
                 requires_grad: bool = False,
                 grad_fn: Callable = None,
                 children: List[Tensor] = None,
                 leaf: bool = True):
        if children is None:
            children = []
        self._data: np.ndarray = convert_to_array(data)
        self._grad: Optional[Tensor] = None
        self._requires_grad = requires_grad
        self._grad_fn = grad_fn
        self._children = children
        self._leaf = leaf
        # np.array attributes
        self._size = self.data.size
        self._shape = self.data.shape
        self._dims = self.data.ndim
        self._datatype = self.data.dtype
        if self._requires_grad:
            self._grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    @classmethod
    def random(cls, shape: Tuple):
        return cls(np.random.rand(shape))

    def reshape(self, dims):
        return Tensor(self.data.reshape(dims))

    def resize(self, dims):
        self.data.resize(dims)
        self.shape = self.data.shape

    def backward(self,
                 grad: Optional[Tensor] = None):
        if self.requires_grad:
            if grad is None:
                if self._dims != 0:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                else:
                    grad = Tensor(1)
            self.grad.data += grad.data
            if not self.leaf:
                gradients = self.grad_fn(self.grad.data)
                for child, gradient in zip(self.children, gradients):
                    if gradient is not None and child.requires_grad is True:
                        child.backward(gradient)
        else:
            raise RuntimeError(".backward() run on Tensor with requires_grad=False")

    def __str__(self):
        return f"Tensor(data={self.data})"

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __eq__(self, other):
        return (np.array_equal(self.data, other.data) and self.size == other.size and
                self.shape == other.shape and self.dims == other.dims and
                self.datatype == other.datatype)

    def _unary_op(self, grad_type):
        grad_fn = grad_type(self.data)
        return Tensor(grad_fn.data,
                      self.requires_grad,
                      grad_fn,
                      [self],
                      False)

    def _binary_op(self, other, grad_type):
        grad_fn = grad_type(self.data, other.data)
        return Tensor(grad_fn.data,
                      (self.requires_grad or other.requires_grad),
                      grad_fn,
                      [self, other],
                      False)

    # primitive ops.
    def sum(self):
        return self._unary_op(Sum)
    def __neg__(self):
        return self._unary_op(Neg)
    def exp(self):
        return self._unary_op(Exp)
    def log(self):
        return self._unary_op(Log)
    def square(self):
        return self._unary_op(Square)
    def sqrt(self):
        return self._unary_op(Sqrt)
    def sin(self):
        return self._unary_op(Sin)
    def cos(self):
        return self._unary_op(Cos)
    def tan(self):
        return self._unary_op(Tan)
    def arcsin(self):
        return self._unary_op(Arcsin)
    def arccos(self):
        return self._unary_op(Arccos)
    def arctanh(self):
        return self._unary_op(Arctan)
    def sinh(self):
        return self._unary_op(Sinh)
    def cosh(self):
        return self._unary_op(Cosh)
    def tanh(self):
        return self._unary_op(Tanh)
    def arcsinh(self):
        return self._unary_op(Arcsinh)
    def arccosh(self):
        return self._unary_op(Arccosh)
    def arctanh(self):
        return self._unary_op(Arctanh)

    def __add__(self, other):
        return self._binary_op(convert_to_operable(other), Add)
    def __sub__(self, other):
        return self._binary_op(convert_to_operable(other), Sub)
    def __mul__(self, other):
        return self._binary_op(convert_to_operable(other), Mul)
    def __truediv__(self, other):
        return self._binary_op(convert_to_operable(other), Div)
    def __matmul__(self, other):
        return self._binary_op(convert_to_operable(other), MatMul)

    # compound ops.
    def sigmoid(self):
        return Tensor(1, self.requires_grad) / (Tensor(1, self.requires_grad) + (-self).exp())

    # attributes
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be of type np.ndarray")
        self._data = data
        self._size = self.data.size
        self._shape = self.data.shape
        self._dims = self.data.ndim
        self._datatype = self.data.dtype

    @property
    def grad(self):
        return self._grad

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def children(self):
        return self._children

    @property
    def leaf(self):
        return self._leaf
