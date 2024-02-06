import numpy as np
import cupy as cp
from core.tensor import Tensor


class TestTensor:
    def test_tensor_1(self):
        t1 = Tensor(np.array([1, 2, 3]))
        t2 = Tensor(np.array([1, 2, 3]))
