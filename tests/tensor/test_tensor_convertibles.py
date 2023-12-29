import pytest
import numpy as np
import cupy as cp
from core.tensor import DEVICE, Tensor


class TestTensorConvertibles:
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

    def test_inconvertible(self):
        X = Tensor([1, 2])
        with pytest.raises(TypeError, match="Unsupported operand type\(s\) for Tensor and <class 'list'>"):
            X + [3, 4]
        with pytest.raises(TypeError, match="Unsupported operand type\(s\) for Tensor and <class 'tuple'>"):
            X + (3, 4)
