import cupy as cp
from core.tensor import Tensor


class TestTensorAutoDevice:
    def test_tensor_device_1(self):
        A = Tensor([1, 2, 3], requires_grad=True, device='cuda')
        B = Tensor([1, 2, 3], requires_grad=True, device='cuda')
        C = (A + B).sum()
        C.backward()
