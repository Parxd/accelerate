import numpy as np
from core.tensor import Tensor


class TestTensorSigmoid:
    def test_sigmoid_1(self):
        X = Tensor(0.5, requires_grad=True)
        Y = X.sigmoid()
        assert np.allclose(Y.data, np.array(0.6224593312018959))
        Y.backward()
        assert np.allclose(X.grad.data, np.array(0.23500371220158436))
