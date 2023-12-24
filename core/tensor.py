from __future__ import annotations
from typing import Any, List, Optional
from enum import Enum
import warnings
import cupy as cp
import numpy as np
from auto.tensor.add import add, AddBackward

ScalarType = int | float
PrimType = ScalarType | List
ArrayType = np.ndarray | cp.ndarray
TensorType = PrimType | ArrayType


def check_data(data: Any):
    if not isinstance(data, TensorType):
        raise TypeError("Tensor must be initialized from int, float, List, np.ndarray, or cp.ndarray")
    if isinstance(data, PrimType):
        return np.array(data, dtype=np.float64)
    return data


class DEVICE(Enum):
    CPU = 0
    GPU = 1


class Tensor:
    def __init__(self,
                 data: TensorType,
                 grad: Optional[Tensor] = None,
                 requires_grad: bool = False,
                 grad_fn=None,
                 children=None,
                 leaf=True):
        if children is None:
            children = []
        self._data: np.ndarray | cp.ndarray = check_data(data)
        self._grad = grad
        self._requires_grad = requires_grad
        self._grad_fn = grad_fn
        self._children = children
        self._leaf = leaf
        # np/cp.array attributes
        self._size = self.data.size
        self._shape = self.data.shape
        self._dims = self.data.ndim
        self._datatype = self.data.dtype
        self._device = DEVICE.CPU if isinstance(self.data, np.ndarray) else DEVICE.GPU

    def reshape(self, dims):
        return Tensor(self.data.reshape(dims))

    def resize(self, dims):
        self.data.resize(dims)
        self.shape = self.data.shape

    def to(self,
           device: DEVICE):
        if self.device == device:
            return
        if device == DEVICE.CPU:
            self._data = self._data.get()
        else:
            self._data = cp.array(self._data)
        self._device = device

    def backward(self,
                 grad: Optional[Tensor] = None):
        if self.requires_grad:
            if grad is None:
                if self.shape != ():
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                else:
                    grad = Tensor(1)
            self.grad.data += grad
            if not self.leaf:
                ...
        else:
            warnings.warn(".backward() called on a Tensor with requires_grad=False")

    def __str__(self):
        return f"Tensor(data={self.data}, device={self.device})"

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __eq__(self, other):
        return ((self.data == other.data).all() and self.size == other.size and
                self.shape == other.shape and self.dims == other.dims and
                self.datatype == other.datatype and self.device == other.device)

    # operations
    def sum(self):
        ...

    def __add__(self, other):
        other = self._convert_to_operable(other)
        self._check_devices(other)

    def _convert_to_operable(self,
                             other):
        if isinstance(other, Tensor):
            return other
        array_type = np.array if self.device == DEVICE.CPU else cp.array
        if isinstance(other, ScalarType):
            return Tensor(array_type(other))
        elif isinstance(other, ArrayType):
            return Tensor(other)
        raise TypeError(f"unsupported operand type(s) for Tensor and {type(other)}")

    def _check_devices(self,
                       other: Tensor):
        if self.device != other.device:
            raise TypeError(f"Tensor device mismatch, found {self.device} and {other.device}")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be of type np.ndarray")
        if self.device == DEVICE.CPU:
            self._data = data
        else:
            self._data = cp.array(data)
        self._size = self.data.size
        self._shape = self.data.shape
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
        # check if Type[Backward]
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
        if self.device == DEVICE.CPU:
            self._datatype = datatype
            self.data.dtype = datatype
        else:
            raise AttributeError("cannot modify datatype of Tensor on GPU")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        raise AttributeError("device is not a writeable attribute; use Tensor.to() instead")
