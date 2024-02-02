import numpy as np
from core.tensor import Tensor
from .layer import Layer


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self._w = Tensor.random((out_features, in_features), requires_grad=True)
        self._b = Tensor.random((1, out_features), requires_grad=True) if bias else Tensor(np.zeros(1, out_features))

    def __call__(self, data: Tensor) -> Tensor:
        return data @ self._w.transpose() + self._b

    def __str__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"

    def parameters(self):
        return [self._w, self._b]

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
