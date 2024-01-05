import numpy as np
import cupy as cp
import pytest
from core.tensor import Tensor, Device


CPU = Device.CPU
GPU = Device.GPU


class TestTensorDevice:
    def test_tensor_device_1(self):
        ...
