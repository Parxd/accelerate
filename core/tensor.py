from __future__ import annotations
from typing import Any, List, Optional
import warnings
import numpy as np

ScalarType = int | float
PrimType = ScalarType | List
TensorType = PrimType | np.ndarray


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


def convert_to_array(data: Any):
    if not isinstance(data, TensorType):
        raise TypeError("Tensor must be initialized from int, float, List, np.ndarray")
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
                 grad_fn=None,
                 children=None,
                 leaf=True):
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

    def reshape(self, dims):
        return Tensor(self.data.reshape(dims))

    def resize(self, dims):
        self.data.resize(dims)
        self.shape = self.data.shape

    def backward(self,
                 grad: Optional[Tensor] = None):
        if self.requires_grad:
            if grad is None:
                if self.dims != 0:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                else:
                    grad = Tensor(1)
            self.grad.data += grad.data
            if not self.leaf:
                gradients = self.grad_fn(self.grad.data)
                for child, gradient in zip(self.children, gradients):
                    if gradient is not None:
                        child.backward(gradient)
        else:
            print(self.data)
            warnings.warn(".backward() called on Tensor with requires_grad=False")
    
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

    # operations
    def sum(self):
        data = self.data.sum()
        def sum_grad(grad):
            return (grad,)
        return Tensor(data,
                      self.requires_grad,
                      sum_grad,
                      [self],
                      False)

    def exp(self):
        data = np.exp(self.data)
        def exp_grad(grad):
            return (grad * data,)
        return Tensor(data,
                      self.requires_grad,
                      exp_grad,
                      [self],
                      False)

    def __neg__(self):
        data = -self.data
        def neg_grad(grad):
            return (-grad,)
        return Tensor(data,
                      self.requires_grad,
                      neg_grad,
                      [self],
                      False)

    def __add__(self, other):
        data = self.data + other.data
        def add_grad(grad):
            return (grad, grad)
        return Tensor(data,
                      self.requires_grad or other.requires_grad,
                      add_grad,
                      [self, other],
                      False)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        data = self.data * other.data
        def mul_grad(grad):
            return (grad * other.data, grad * self.data)
        return Tensor(data,
                      self.requires_grad or other.requires_grad,
                      mul_grad,
                      [self, other],
                      False)

    def __truediv__(self, other):
        data = self.data / other.data
        def div_grad(grad):
            return (grad / other.data, -grad * self.data / other.data ** 2)
        return Tensor(data,
                      self.requires_grad or other.requires_grad,
                      div_grad,
                      [self, other],
                      False)

    def sigmoid(self):
        return Tensor(1) / (Tensor(1) + (-self).exp())

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

    @grad.setter
    def grad(self, grad):
        if not isinstance(grad, Tensor):
            raise TypeError("grad must be of type Tensor")
        warnings.warn("modifying grad will likely break auto-differentiation")
        self._grad = grad

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be of type bool")
        self._requires_grad = requires_grad

    @property
    def grad_fn(self):
        return self._grad_fn

    @grad_fn.setter
    def grad_fn(self, grad_fn):
        warnings.warn("modifying grad_fn will likely break auto-differentiation")
        self._grad_fn = grad_fn

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        if not isinstance(children, List):
            raise TypeError("children must be of type List[Tensor]")
        warnings.warn("modifying children will likely break auto-differentiation")
        self._children = children

    @property
    def leaf(self):
        return self._leaf

    # np/cp.array attributes
    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        raise AttributeError("size is not a writeable attribute")

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.data.shape = shape

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, dims):
        raise AttributeError("dims is not a writeable attribute")

    @property
    def datatype(self):
        return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        self._datatype = datatype
        self.data.dtype = datatype
