from core.tensor import Tensor
from .module import Module


class Sigmoid(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        return x.sigmoid()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __str__(self):
        return "Sigmoid()"
