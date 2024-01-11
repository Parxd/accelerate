import numpy as np
from core.tensor import Tensor
from nn.loss import *


class TestBCELoss:
    def test_bce_1(self):
        predicted = Tensor(0.5)
        truth = Tensor(1.0)
        loss = BCELoss(reduction='sum')
        assert np.isclose(loss(predicted, truth).data, np.array(0.6931), rtol=1e-4)
