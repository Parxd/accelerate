import numpy as np
from core.tensor import Tensor
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.w = Tensor(np.random.randn(out_features, in_features), requires_grad=True)
        self.b = Tensor(np.random.randn(1, out_features), requires_grad=True) if bias \
            else Tensor(np.zeros(1, out_features))
        super().__init__(
            [
                self.w,
                self.b
            ]
        )
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def forward(self, x: Tensor):
        return x @ self.w.transpose() + self.b

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __str__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"
