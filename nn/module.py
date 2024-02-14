from abc import ABC, abstractmethod
from core.tensor import Tensor


class Module(ABC):
    def __init__(self,
                 parameters=None) -> None:
        if parameters is None:
            parameters = []
        self._parameters = parameters

    def parameters(self):
        # recursive traversal
        for param in self._parameters:
            # non-leaf node
            if isinstance(param, Module):
                yield from param.parameters()
            # leaf node
            else:
                yield param

    def named_parameters(self):
        ...

    def zero_grad(self):
        for layer in self._parameters:
            layer.zero_grad()

    def to(self, device: str):
        for tensor in self.parameters():
            tensor.to(device)

    @abstractmethod
    def forward(self, x: Tensor):
        return NotImplementedError("forward must be defined by derived class")
