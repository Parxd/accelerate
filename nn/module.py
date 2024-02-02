from abc import ABC, abstractmethod
from typing import List
from core.tensor import Tensor


class Module(ABC):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        self._children = []
        self._parameters = []
        for arg in args:
            self._parameters.append(arg)

    def children(self):
        ...

    def parameters(self):
        # recursive traversal
        for param in self._parameters:
            # non-leaf node
            if not isinstance(param, Tensor):
                yield from param.parameters()
            # leaf node
            else:
                yield f"Parameter containing:\n{param}, requires_grad={param.requires_grad}"

    def named_parameters(self):
        ...

    @abstractmethod
    def forward(self, x: Tensor):
        return NotImplementedError("forward method must be defined by derived class")

    def zero_grad(self):
        for layer in self._children:
            layer.zero_grad()
