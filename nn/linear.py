import numpy as np
from core.tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self._w = Tensor.random((out_features, in_features), requires_grad=True)
        self._b = Tensor.random((1, out_features), requires_grad=True) if bias else Tensor(np.zeros(1, out_features))

    def __call__(self, data: Tensor) -> Tensor:
        return data @ self._w.transpose() + self._b
