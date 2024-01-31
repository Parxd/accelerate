import numpy as np
from core import *
from nn.loss import MSELoss


class TestTensorNoGrad:
    def test_nograd_1(self):
        x = Tensor([2], requires_grad=True)

        criterion = MSELoss()
        loss = criterion(x, Tensor([0.5]))
        loss.backward()

        with NoGrad():
            x -= x.grad
