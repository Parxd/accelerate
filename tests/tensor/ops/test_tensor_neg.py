import numpy as np
from core.tensor import Tensor


class TestTensorNeg:
    def test_neg_1(self):
        X = Tensor(5, requires_grad=True)
        Y = -X
        assert np.array_equal(Y.data, np.array(-5))
        Y.backward()
        assert np.array_equal(X.grad.data, np.array(-1))
