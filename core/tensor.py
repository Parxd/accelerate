from __future__ import annotations
from enum import Enum
import cupy as cp
import numpy as np

TensorType = np.ndarray | cp.ndarray


def ensure_type(tensor: TensorType, device: DEVICE):
    if device == DEVICE.GPU and not isinstance(tensor, cp.ndarray) or \
            device == DEVICE.CPU and not isinstance(tensor, np.ndarray):
        raise TypeError("device type not aligned with data")


def device_check(tensor: TensorType):
    if not isinstance(tensor, TensorType):
        raise TypeError("array not of TensorType")
    if isinstance(tensor, np.ndarray):
        return DEVICE.CPU
    else:
        return DEVICE.GPU


class DEVICE(Enum):
    CPU = 0
    GPU = 1


class Tensor:
    """
    Tensor class
    - either lives on CPU or GPU
    - supports multiple datatypes
    - needs to check if type(tensor.data) aligns with tensor.device
    """
    def __init__(self,
                 data,
                 requires_grad=False,
                 grad=None,
                 grad_fn=None,
                 children=None,
                 leaf=True,
                 datatype=np.float32):
        if children is None:
            children = []
        self.data = data
        self.requires_grad = requires_grad
        self.grad = grad
        self.grad_fn = grad_fn
        self.children = children
        self.leaf = leaf
        self.device = device_check(self.data)
        self.datatype = datatype

    def set_device(self, device):
        if self.device == device:
            return
        if device == DEVICE.CPU:
            self.data = np.asarray(self.data)
        else:
            self.data = cp.asarray(self.data)
        self.device = device
