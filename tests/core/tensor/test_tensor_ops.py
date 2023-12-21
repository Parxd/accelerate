import numpy as np
from core.tensor import DEVICE, Tensor


class TestTensorOps:
    def test_add(self):
        a = Tensor([[1, 2, 3],
                    [4, 5, 6]])
        b = Tensor([[7, 8, 9],
                    [10, 11, 12]])
        c = a + b
        assert c.size == 6
        assert c.shape == (2, 3)
        assert c.data.shape == (2, 3)
        assert c.dims == 2
        assert c.datatype == np.float32
        assert c.device == DEVICE.CPU

        assert c == Tensor([[8, 10, 12],
                            [14, 16, 18]])
        # gradients
