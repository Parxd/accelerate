import numpy as np
from core.tensor import Tensor


class TestTensorSum:
    def test_sum(self):
        X = Tensor([[1, 2], [3, 4]], requires_grad=True)
        Y = X.sum()
        assert np.array_equal(Y.data, np.array(10))

        Y.backward()
        assert np.array_equal(X.grad.data, np.array([[1, 1],
                                                     [1, 1]], dtype=np.float64))
    
    