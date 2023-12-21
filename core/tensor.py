from __future__ import annotations
from typing import Any, List
from enum import Enum
import cupy as cp
import numpy as np
from auto.math.tensor.add import add

ScalarType = int | float
PrimType = ScalarType | List
ArrayType = np.ndarray | cp.ndarray
TensorType = np.ndarray | cp.ndarray | PrimType


def check_data(data: Any):
    if not isinstance(data, TensorType):
        raise TypeError("Tensor must be initialized from int, float, List, np.ndarray, or cp.ndarray")
    if isinstance(data, PrimType):
        return np.array(data)
    return data


class DEVICE(Enum):
    CPU = 0
    GPU = 1


class Tensor:
    def __init__(self,
                 data: TensorType,
                 grad=0,
                 requires_grad=False,
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

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        raise AttributeError("data is not a writeable attribute")

    # np/cp.array attribute getters/setters
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

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        raise AttributeError("device is not a writeable attribute; use Tensor.to() instead")

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

    def __add__(self, other):
        other = self._convert_to_operable(other)
        self._check_devices(other)
        return Tensor(add(self.data, other.data))

    def _convert_to_operable(self,
                             other):
        if isinstance(other, Tensor):
            return other
        array_type = np.array if self.device == DEVICE.CPU else cp.array
        if isinstance(other, ScalarType):
            return Tensor(array_type(other))
        elif isinstance(other, ArrayType):
            return Tensor(other)
        raise TypeError(f"Unsupported operand type(s) for Tensor and {type(other)}")

    def _check_devices(self,
                       other: Tensor):
        if self.device != other.device:
            raise TypeError(f"Tensor device mismatch, found {self.device} and {other.device}")
