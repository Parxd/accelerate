import numpy as np
import cupy as cp
from core.tensor import Tensor, CPU, GPU


class TestTensor:
    def test_tensor_1(self):
        # Test CPU construction
        T1 = Tensor(np.array([1, 2, 3]))
        T2 = Tensor(cp.array([1, 2, 3]))
        assert isinstance(T1.data, np.ndarray)
        assert isinstance(T2.data, np.ndarray)

    def test_tensor_2(self):
        # Test GPU construction
        T1 = Tensor(np.array([1, 2, 3]), device="cuda")
        T2 = Tensor(cp.array([1, 2, 3]), device="cuda")
        assert isinstance(T1.data, cp.ndarray)
        assert isinstance(T2.data, cp.ndarray)

    def test_tensor_3(self):
        # Test Tensor.to() method
        T1 = Tensor(np.array([1, 2, 3]))
        T1.to("cuda")
        assert T1.device == GPU
        assert isinstance(T1.data, cp.ndarray)
        T1.to("cpu")
        assert T1.device == CPU
        assert isinstance(T1.data, np.ndarray)

    def test_tensor_4(self):
        # Test CPU & GPU construction from non-np/cp array types
        Tensor(5)
        Tensor(5.)
        Tensor([5.])
        Tensor(5, device="cuda")
        Tensor(5., device="cuda")
        Tensor([5.], device="cuda")
