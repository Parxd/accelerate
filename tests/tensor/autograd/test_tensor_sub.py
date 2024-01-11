import numpy as np
from core.tensor import Tensor


class TestTensorSub:
    def test_sub_1(self):
        X = Tensor([1, 2, 3])
        Y = Tensor([4, 5, 6])
        Z = Y - X
        print(Z)

    def test_sub_2(self):
        X = Tensor([5, 5, 5])
        Y = 6 - X
        print(Y)
