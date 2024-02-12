import numpy as np
from core.tensor import Tensor


class TestTensorDetachClone:
    def test_tensor_detach_1(self):
        A = Tensor(np.array([1, 2, 3]), requires_grad=True)
        B = Tensor(np.array([1, 2, 3]))
        C = (A + B).sum()
        A = A.detach()
        C.backward()
        # check that they share same memory
        A_detached = A.detach()
        assert A_detached.data is A.data
        # gradient should not have been calculated
        assert A.grad is None

    def test_tensor_detach_2(self):
        A = Tensor(np.array([1, 1]), requires_grad=True)
        B = Tensor(np.array([1, 1]))
        C = Tensor(np.array([1, 1]), requires_grad=True)
        D = Tensor(np.array([1, 1]))
        E = A + B
        F = C * D
        G = (E + F).sum()
        E = E.detach()
        G.backward()
        assert A.grad is not None
        assert E.grad is None

    def test_tensor_detach_3(self):
        X = Tensor(np.array([1, 2, 3]))
        Y = X.detach()
        Y[0] = 0
        # test shallow mem-copy
        assert X.data is Y.data
        assert X == Y

    def test_tensor_clone_1(self):
        X = Tensor(np.array([1, 2, 3]), requires_grad=True)
        Y = X.clone()
        Z = Y.square().sum()
        Z.backward()
        assert X.grad is not None
        assert Y.grad is not None
        assert X.grad == Y.grad

    def test_tensor_clone_2(self):
        X = Tensor(np.array([1, 2, 3]))
        Y = X.clone()
        Y[0] = 0
        # test deep mem-copy
        assert X.data is not Y.data
        assert X != Y
