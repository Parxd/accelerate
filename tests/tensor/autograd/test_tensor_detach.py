import numpy as np
from core.tensor import Tensor


class TestTensorDetach:
    def test_tensor_detach_1(self):
        A = Tensor(np.array([1, 2, 3]), requires_grad=True)
        B = Tensor(np.array([1, 2, 3]), requires_grad=True)
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
        B = Tensor(np.array([1, 1]), requires_grad=True)
        C = Tensor(np.array([1, 1]), requires_grad=True)
        D = Tensor(np.array([1, 1]), requires_grad=True)
        E = A + B
        F = C * D
        G = (E + F).sum()
        E = E.detach()
        G.backward()
        # child of
        assert A.grad is not None
        assert E.grad is None
