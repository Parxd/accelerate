import numpy as np
from core.tensor import Tensor
from nn.loss import *


class TestBCELoss:
    def test_bce_1(self):
        predicted = Tensor(0.99999999999999999)
        truth = Tensor(1.0)
        loss = BCELoss()
        assert np.isclose(loss(predicted, truth).data, np.array(0.0000), rtol=1e-3)

    def test_bce_2(self):
        predicted = Tensor(0.5)
        truth = Tensor(1.0)
        loss = BCELoss()
        assert np.isclose(loss(predicted, truth).data, np.array(0.6931), rtol=1e-4)

    def test_bce_3(self):
        predicted = Tensor(0.5, requires_grad=True)
        truth = Tensor(1.0)
        loss = BCELoss()
        error = loss(predicted, truth)
        error.backward()
        print(predicted.grad)
        # assert np.isclose(predicted.grad.data, np.array(-2), rtol=1e-4)
