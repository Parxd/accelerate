import pytest
import numpy as np
import cupy as cp
from core.tensor import DEVICE, Tensor


CPU = DEVICE.CPU
GPU = DEVICE.GPU


class TestTensorDevice:
    def test_check_data(self):
        X = Tensor([1, 2, 3])
        Y = Tensor(np.array([1, 2, 3]))
        Z = Tensor(cp.array([1, 2, 3]))
        assert X.device == CPU
        assert Y.device == CPU
        assert Z.device == GPU

        with pytest.raises(TypeError):
            Tensor("1, 2, 3")
            Tensor(True)
            Tensor((1, 2, 3))
            Tensor({})

    def test_to_different(self):
        X = Tensor([1, 2, 3])
        X.to(GPU)
        assert X.device == GPU
        assert isinstance(X.data, cp.ndarray)

        X.to(CPU)
        assert X.device == CPU
        assert isinstance(X.data, np.ndarray)
        assert (X.data == np.array([1, 2, 3])).all()

    def test_to_same(self):
        X = Tensor([1, 2, 3])
        assert X.device == CPU
        X.to(CPU)
        assert X.device == CPU
        assert isinstance(X.data, np.ndarray)
