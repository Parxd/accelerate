import pytest
import numpy as np
import cupy as cp
from core.tensor import DEVICE, Tensor


class TestTensorBasicOps:
    def test_scalar_convert_to_operable_cpu(self):
        a = Tensor([1, 2])
        b = 3
        assert (a + b) == Tensor([4, 5])

    def test_scalar_convert_to_operable_gpu(self):
        X = Tensor([1, 2])
        X.to(DEVICE.GPU)
        y = 3
        X + y

    def test_array_convert_to_operable_cpu(self):
        X = Tensor([1, 2])
        Y = np.array([3, 4])
        X + Y
        with pytest.raises(TypeError, match="Tensor device mismatch, found DEVICE.CPU and DEVICE.GPU"):
            Z = cp.array([3, 4])
            X + Z

    def test_array_convert_to_operable_gpu(self):
        X = Tensor([1, 2])
        X.to(DEVICE.GPU)
        Y = cp.array([3, 4])
        X + Y
        with pytest.raises(TypeError, match="Tensor device mismatch, found DEVICE.GPU and DEVICE.CPU"):
            Z = np.array([3, 4])
            X + Z

    def test_add(self):
        a = Tensor([[1, 2, 3],
                    [4, 5, 6]])
        b = Tensor([[7, 8, 9],
                    [10, 11, 12]])
        c = a + b
        assert c.size == 6
        assert c.shape == (2, 3)
        assert c.data.shape == (2, 3)
        assert c.dims == 2
        assert c.datatype == np.float32
        assert c.device == DEVICE.CPU

        assert c == Tensor([[8, 10, 12],
                            [14, 16, 18]])
        # gradients
