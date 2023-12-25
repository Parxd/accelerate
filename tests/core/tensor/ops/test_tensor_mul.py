import numpy as np
from core.tensor import Tensor


class TestTensorMul:
    def test_mul_1(self):  # test backward w/ 0-dim tensor operands
        X = Tensor(5, requires_grad=True)
        Y = Tensor(10, requires_grad=True)
        Z = X * Y
        assert Z.data == np.array(50)
        Z.backward()
        assert X.grad.data == np.array(10)
        assert Y.grad.data == np.array(5)

    def test_mul_2(self):  # test backward without given gradient
        X = Tensor([[1, 2],
                    [3, 4]], requires_grad=True)
        Y = Tensor([[1, 2],
                    [3, 4]], requires_grad=True)
        Z = (X * Y).sum()
        Z.backward(Tensor([[1, 1],
                           [1, 1]]))
        assert Z.data == np.array([[1, 4]],
                                  [9, 16])
        assert np.array_equal(X.grad.data, np.array([]))
        assert np.array_equal(Y.grad.data, np.array([]))

    def test_mul_3(self):
        ...
