import numpy as np
from core.tensor import Tensor
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            [
                Tensor.random((out_features, in_features), requires_grad=True),
                Tensor.random((1, out_features), requires_grad=True) if bias else Tensor(np.zeros(1, out_features))
            ]
        )
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def forward(self, x: Tensor):
        return x @ self._parameters[0].transpose() + self._parameters[1]

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __str__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"
