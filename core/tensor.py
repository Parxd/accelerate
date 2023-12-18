from __future__ import annotations
from typing import Type, List
from enum import Enum
import cupy as cp
import numpy as np

TensorConstructType = np.ndarray | List
TensorType = TensorConstructType | cp.ndarray


def check_tensor(tensor):
    if not isinstance(tensor, TensorConstructType):
        raise TypeError("tensor must be constructed from list or numpy array")
    if isinstance(tensor, List):
        return np.array(tensor)
    return tensor


class DEVICE(Enum):
    CPU = 0
    GPU = 1


class Tensor:
    def __init__(self,
                 data: TensorType):
        self.data = check_tensor(data)
        self.shape = self.data.shape
        self.device = DEVICE.CPU

    def to(self,
           device: DEVICE):
        # Loads data to specified device type
        self._check_alignment()
        if self.device == device:
            return
        if device == DEVICE.CPU:
            self.data = self.data.get()
        else:
            self.data = cp.array(self.data)
        self.device = device

    def __str__(self):
        return f"Tensor(data={self.data}, device={self.device})"

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def _check_alignment(self):
        if self.device == DEVICE.CPU and not isinstance(self.data, np.ndarray) or \
                self.device == DEVICE.GPU and not isinstance(self.data, cp.ndarray):
            raise TypeError("data and device not aligned")
