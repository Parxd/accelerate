import numpy as np
from core.tensor import Tensor


class TestTensorAdd:
    def test_add_1(self):  # test backward w/ 0-dim tensor operands
        X = Tensor(15, requires_grad=True)
        Y = Tensor(60, requires_grad=True)
        Z = X + Y
        Z.backward()
        assert Z.data == np.array(75)
        assert np.array_equal(X.grad.data, np.array(1))
        assert np.array_equal(Y.grad.data, np.array(1))

    def test_add_2(self):  # test backward without given gradient
        X = Tensor([[1, 2],
                    [3, 4]], requires_grad=True)
        Y = Tensor([[1, 2],
                    [3, 4]], requires_grad=True)
        Z = (X + Y).sum()
        Z.backward()
        assert np.array_equal(X.grad.data, np.array([[1, 1],
                                                     [1, 1]]))
        assert np.array_equal(Y.grad.data, np.array([[1, 1],
                                                     [1, 1]]))

    def test_add_3(self):  # test backward with given gradient
        X = Tensor([[1, 2],
                    [3, 4]], requires_grad=True)
        Y = Tensor([[1, 2],
                    [3, 4]], requires_grad=True)
        Z = X + Y
        Z.backward(Tensor([[2, 2],
                           [2, 2]]))
        assert np.array_equal(X.grad.data, np.array([[2, 2],
                                                     [2, 2]]))
        assert np.array_equal(Y.grad.data, np.array([[2, 2],
                                                     [2, 2]]))

    def test_add_4(self):  # test tensors without grad requirement do not have gradient
        X = Tensor([[1, 2],
                    [3, 4]], requires_grad=True)
        Y = Tensor([[1, 2],
                    [3, 4]])  # Y.requires_grad == False
        Z = (X + Y).sum()
        Z.backward()
        assert X.grad is not None
        assert Y.grad is None

    def test_add_5(self):  # test backward w/ broadcasting involved pt. 1
        X = Tensor(2, requires_grad=True)
        Y = Tensor([[0, 2],
                    [4, 6]], requires_grad=True)
        Z = (X + Y).sum()
        Z.backward()
        assert Z.data == np.array(20)
        assert np.array_equal(X.grad.data, np.array(4))
        assert np.array_equal(Y.grad.data, np.array([[1, 1],
                                                     [1, 1]]))

    def test_add_6(self):  # test backward w/ broadcasting involved pt. 2
        X = Tensor([2, 5], requires_grad=True)
        Y = Tensor([[0, 2],
                    [4, 6]], requires_grad=True)
        Z = (X + Y).sum()
        Z.backward()
        assert Z.data == np.array(26)
        assert np.array_equal(X.grad.data, np.array([2, 2]))
        assert np.array_equal(Y.grad.data, np.array([[1, 1],
                                                     [1, 1]]))

    def test_add_7(self):  # test backward w/ broadcasting involved pt. 3
        X = Tensor([1, 2, 3], requires_grad=True)
        Y = Tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], requires_grad=True)
        Z = (X + Y).sum()
        assert Z.data == np.array(63)
        assert np.array_equal(X.grad.data, np.array([3, 3, 3]))
        assert np.array_equal(Y.grad.data, np.ones_like(Y.grad.data))
