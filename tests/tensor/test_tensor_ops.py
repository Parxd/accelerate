import numpy as np
import cupy as cp
from core.tensor import Tensor, tensor_conv


class TestTensorOps:
    def test_tensor_conv(self):
        A = Tensor([1, 2, 3])
        assert tensor_conv(A, A.device) is A
        assert tensor_conv([1.0, 2.0, 3.0], A.device) == A

    def test_tensor_with_int_cpu(self):
        A = Tensor([1, 2, 3])
        B = A + 5
        assert B == Tensor([6, 7, 8])

        B = A - 1
        assert B == Tensor([0, 1, 2])

        B = A * 10
        assert B == Tensor([10, 20, 30])

        B = A / 3
        assert np.allclose(B.data, np.array([0.333333, 0.666666, 1]))

    def test_tensor_with_reverse_int_cpu(self):
        A = Tensor([1, 2, 3])
        B = 5 + A
        assert B == Tensor([6, 7, 8])

        B = 1 - A
        assert B == Tensor([0, -1, -2])

        B = 10 * A
        assert B == Tensor([10, 20, 30])

        B = 1 / A
        assert np.allclose(B.data, np.array([1, 0.5, 0.333333]))

    def test_tensor_with_int_gpu(self):
        A = Tensor([1, 2, 3], device='cuda')
        B = A + 5
        assert B == Tensor([6, 7, 8], device='cuda')

        B = A - 1
        assert B == Tensor([0, 1, 2], device='cuda')

        B = A * 10
        assert B == Tensor([10, 20, 30], device='cuda')

        B = A / 3
        assert cp.allclose(B.data, np.array([0.333333, 0.666666, 1]))

    def test_tensor_with_reverse_int_gpu(self):
        A = Tensor([1, 2, 3], device='cuda')
        B = 5 + A
        assert B == Tensor([6, 7, 8], device='cuda')

        B = 1 - A
        assert B == Tensor([0, -1, -2], device='cuda')

        B = 10 * A
        assert B == Tensor([10, 20, 30], device='cuda')

        B = 1 / A
        assert cp.allclose(B.data, np.array([1, 0.5, 0.333333]))

    def test_tensor_with_numpy(self):
        A = Tensor([1, 2, 3])
        B = np.array([4, 5, 6])

        C = A - B
        assert C == Tensor([-3, -3, -3])

        # need to fix by overloading ufunc for np arrays
        C = B - A
        assert isinstance(C, np.ndarray)

        C = A * B
        assert C == Tensor([4, 10, 18])

        # need to fix by overloading ufunc for np arrays
        C = B * A
        assert isinstance(C, np.ndarray)
