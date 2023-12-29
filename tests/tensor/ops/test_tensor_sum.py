import numpy as np
from core.tensor import Tensor


class TestTensorSum:
    def test_sum_1(self):
        X = Tensor([[1, 2],
                    [3, 4]], requires_grad=True)
        Y = X.sum()
        assert np.array_equal(Y.data, np.array(10))

        Y.backward()
        assert np.array_equal(X.grad.data, np.array([[1, 1],
                                                     [1, 1]], dtype=np.float64))

    def test_sum_2(self):
        X = Tensor([1, 2, 3, 4], requires_grad=True)
        Y = X.sum()
        Y.backward(Tensor(2))

    def test_sum_3(self):
        X = Tensor(1)
        Y = X.sum()
        assert Y.data == np.array(1)
        assert X.grad is None
