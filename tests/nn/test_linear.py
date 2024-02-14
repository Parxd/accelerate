import numpy as np
import cupy as cp
from core.tensor import Tensor
from core.device import Device
from nn import *


class TestLinear:
    def test_linear_1(self):
        layer = Linear(3, 2)
        assert layer.in_features == 2
        assert layer.out_features == 3
        assert layer.bias is True

    def test_linear_2(self):
        layer = Linear(3, 2)
        # data is in shape (FEATURES, DATA_POINTS)
        # example: 6 data points
        X = Tensor(np.array([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]]))
        fwd = layer(X)
        assert isinstance(fwd, Tensor)
        assert fwd.shape == (6, 2)

    def test_linear_3(self):
        layer = Linear(3, 5)
        for tensor in layer.parameters():
            tensor.to('cuda')
            assert tensor.device is Device.GPU
            assert isinstance(tensor.data, cp.ndarray)
        X = Tensor(np.array([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]]), device='cuda')
        fwd = layer(X)
        assert isinstance(fwd, Tensor)
        assert fwd.shape == (6, 5)
