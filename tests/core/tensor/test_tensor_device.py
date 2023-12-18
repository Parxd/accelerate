import pytest
import numpy as np
import cupy as cp
from core.tensor import DEVICE, Tensor


CPU = DEVICE.CPU
GPU = DEVICE.GPU


class TestTensorDevice:
    def test_device_init(self):
        gpu_array = cp.ndarray((1, 2))
        with pytest.raises(TypeError):
            Tensor(gpu_array)

        cpu_array = np.ndarray((1, 2))
        a = Tensor(cpu_array)
        assert isinstance(a, Tensor)
        assert isinstance(a.data, np.ndarray)

        lst = [[1, 2, 3], [4, 5, 6]]
        a = Tensor(lst)
        assert isinstance(a, Tensor)
        assert isinstance(a.data, np.ndarray)

    def test_device_switch(self):
        a = Tensor(np.ndarray((1, 2)))
        a.to(GPU)
        assert a.device == GPU
        assert isinstance(a.data, cp.ndarray)

        a.to(CPU)
        assert a.device == CPU
        assert isinstance(a.data, np.ndarray)

    def test_indexing(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        assert a[0, 0] == 1
        assert a[0, 1] == 2
        assert a[0, 2] == 3
        assert a[1, 0] == 4
        assert a[1, 1] == 5
        assert a[1, 2] == 6
        assert (a[0, :] == np.array([1, 2, 3])).all()
        assert (a[1, :] == np.array([4, 5, 6])).all()
        a[0, 0] = 0
        assert a[0, 0] == 0
