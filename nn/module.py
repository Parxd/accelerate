from abc import ABC, abstractmethod
from core.tensor import Tensor


class Module(ABC):
    def __init__(self,
                 children=None) -> None:
        if children is None:
            children = []
        self._children = children

    def children(self):
        for child in self._children:
            yield child

    def parameters(self):
        # recursive traversal
        for child in self._children:
            # non-leaf node
            if isinstance(child, Module):
                yield from child.parameters()
            # leaf node
            else:
                yield f"Parameter containing:\n{child}, requires_grad={child.requires_grad}"

    def named_parameters(self):
        ...

    @abstractmethod
    def forward(self, x: Tensor):
        return NotImplementedError("forward must be defined by derived class")

    def zero_grad(self):
        for layer in self._children:
            layer.zero_grad()
