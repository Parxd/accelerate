import numpy as np
import cupy as cp
from core.tensorimpl import TensorCPUBackend, TensorGPUBackend


class TestTensorImpl:
    def test_tensor_impl_1(self):
        inst = TensorCPUBackend(np.array([1, 2, 3]), requires_grad=True)
        inst.zero_grad()
        inst.backward()

    def test_tensor_impl_2(self):
        cpu = TensorCPUBackend(np.array([1, 2, 3]), requires_grad=True)
        gpu = TensorGPUBackend(cp.array([1, 2, 3]), requires_grad=True)
        assert not isinstance(cpu, TensorGPUBackend)
        assert not isinstance(gpu, TensorCPUBackend)

    def test_tensor_impl_3(self):
        cpu_int = TensorCPUBackend(1)
        cpu_float = TensorCPUBackend(2.5)
        cpu_list = TensorCPUBackend([1, 2, 3])
        cpu_np = TensorCPUBackend(np.array([1, 2, 3]))

        assert isinstance(cpu_int.data, np.ndarray)
        assert isinstance(cpu_float.data, np.ndarray)
        assert isinstance(cpu_list.data, np.ndarray)
        assert isinstance(cpu_np.data, np.ndarray)

    def test_tensor_impl_4(self):
        gpu_int = TensorGPUBackend(1)
        gpu_float = TensorGPUBackend(2.5)
        gpu_list = TensorGPUBackend([1, 2, 3])
        gpu_np = TensorGPUBackend(np.array([1, 2, 3]))

        assert isinstance(gpu_int.data, cp.ndarray)
        assert isinstance(gpu_float.data, cp.ndarray)
        assert isinstance(gpu_list.data, cp.ndarray)
        assert isinstance(gpu_np.data, cp.ndarray)

    def test_tensor_impl_5(self):
        cpu_rand = TensorCPUBackend.random((3, 5))
        assert isinstance(cpu_rand.data, np.ndarray)
        assert cpu_rand.requires_grad is False

    def test_tensor_impl_6(self):
        t1 = TensorCPUBackend.random((3, 5))
        t2 = TensorCPUBackend.random((3, 5), requires_grad=True)
        t3 = t1 + t2
        assert isinstance(t3, TensorCPUBackend)
        assert isinstance(t3.data, np.ndarray)
        assert t3.requires_grad is True

    def test_tensor_impl_7(self):
        t1 = TensorGPUBackend.random((3, 5))
        t2 = TensorGPUBackend.random((3, 5), requires_grad=True)
        t3 = t1 + t2
        assert isinstance(t3, TensorGPUBackend)
        assert isinstance(t3.data, cp.ndarray)
        assert t3.requires_grad is True

    def test_tensor_impl_8(self):
        t1 = TensorCPUBackend.random((1, 2))
        t2 = t1 + [6, 5]
        assert isinstance(t2, TensorCPUBackend)
        assert isinstance(t2.data, np.ndarray)
