import numpy as np

from core.tensor import Tensor
from nn import *
from nn.loss import *


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
        assert fwd._shape == (6, 2)

    def test_linear_3(self):
        layer = Linear(3, 2)
        X = Tensor([[1, 2, 3],
                    [1, 2, 3]])
        y = Tensor([[0.5, 0.6],
                    [0.7, 0.8]])
        y_hat = layer(X)
        criterion = L1Loss()
        err = criterion(y_hat, y)
        err.backward()

        # ensure gradients are filled out
        assert layer._w.grad != Tensor(np.zeros_like(layer._w.grad))
        assert layer._b.grad != Tensor(np.zeros_like(layer._b.grad))
