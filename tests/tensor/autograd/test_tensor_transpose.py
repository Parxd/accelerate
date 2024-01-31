import numpy as np
from core.tensor import Tensor


class TestTensorTranspose:
    def test_transpose_1(self):
        X = Tensor([[1, 2, 3],
                    [4, 5, 6]])
        Y = X.transpose()

        assert isinstance(Y, Tensor)

        assert X.data.shape == (2, 3)
        assert X._shape == (2, 3)

        assert Y.data.shape == (3, 2)
        assert Y._shape == (3, 2)
