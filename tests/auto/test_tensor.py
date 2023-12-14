import pytest
import numpy as np
import cupy as cp
from core.tensor import device_check, DEVICE, Tensor, TensorType


class TestTensor:
    def test_device_check(self):
        cpu_array = np.ndarray([1, 2])
        assert device_check(cpu_array) == DEVICE.CPU
        gpu_array_1 = cp.ndarray([1, 2])
        assert device_check(gpu_array_1) == DEVICE.GPU
        gpu_array_2 = cp.array(cpu_array)
        assert device_check(gpu_array_2) == DEVICE.GPU

    def test_device_init(self):
        pass