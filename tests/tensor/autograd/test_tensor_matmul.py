import numpy as np
from core.tensor import Tensor


class TestTensorMatMul:
    def test_matmul_1(self):
        X = Tensor([1, 2], requires_grad=True)
        Y = Tensor([[1],
                    [2]], requires_grad=True)
        Z = X @ Y
        assert np.array_equal(Z.data, np.array(5))
        Z.backward()
        assert np.array_equal(X.grad.data, np.array([1, 2]))

    def test_matmul_2(self):
        ...

    def test_matmul_3(self):
        ...
