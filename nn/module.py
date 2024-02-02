from abc import ABC, abstractmethod
from typing import List
from core.tensor import Tensor
from .layer import Layer


class Module(ABC):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        self._children: List[type[Layer]] = []
        self._parameters: List[Tensor] = []
        for arg in args:
            self._children.append(arg)
        for layer in self._children:
            for param in layer.parameters():
                self._parameters.append(param)

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, item: int) -> type[Layer]:
        return self._children[item]

    def children(self):
        for child in self._children:
            yield child

    def parameters(self):
        for param in self._parameters:
            yield f"Parameter containing:\n{param}, requires_grad={param.requires_grad})"

    def named_parameters(self):
        for i, child in enumerate(self._children):
            for param in child.parameters():
                yield f"{i}\nParameter containing:\n{param}, requires_grad={param.requires_grad})"

    @abstractmethod
    def forward(self, x: Tensor):
        return NotImplementedError("forward method must be defined by derived class")

    def zero_grad(self):
        for layer in self._children:
            layer.zero_grad()
