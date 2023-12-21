import pytest
import numpy as np
import cupy as cp
from core.tensor import DEVICE, Tensor


class TestTensorMethods:
    def test_indexing(self):
        a = Tensor([[1, 2, 3],
                    [4, 5, 6]])
        assert a[0, 0] == 1
        assert a[0, 1] == 2
        assert a[0, 2] == 3
        assert a[1, 0] == 4
        assert a[1, 1] == 5
        assert a[1, 2] == 6
        assert (a[0, :] == np.array([1, 2, 3])).all()
        assert (a[1, :] == np.array([4, 5, 6])).all()
        a[0, 0] = 0
        assert a[0, 0] == 0

    def test_resize_inplace(self):
        a = Tensor([[1, 2, 3],
                    [4, 5, 6]])
        a.resize((3, 2))
        assert a.shape == (3, 2)
        assert a.data.shape == (3, 2)

    def test_reshape_inplace(self):
        a = Tensor([[1, 2, 3],
                    [4, 5, 6]])
        a.shape = (3, 2)
        assert a.shape == (3, 2)
        assert a.data.shape == (3, 2)

    def test_reshape_copy(self):
        a = Tensor([[1, 2, 3],
                    [4, 5, 6]])
        b = a.reshape((3, 2))
        assert b.shape == (3, 2)
        assert b.data.shape == (3, 2)

    def test_unwriteables(self):
        # Tensor.data, size, dims, device unwriteable attributes
        a = Tensor([1, 2, 3])
        with pytest.raises(AttributeError):
            a.data = [4, 5, 6]
        with pytest.raises(AttributeError):
            a.size = 6
        with pytest.raises(AttributeError):
            a.dims = 2
        with pytest.raises(AttributeError):
            a.device = DEVICE.GPU
