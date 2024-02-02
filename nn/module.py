from abc import ABC, abstractmethod
from typing import List
from core.tensor import Tensor
from .layer import Layer


class Module(ABC):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        self._layers: List[type[Layer]] = []
        self._parameters: List[Tensor] = []
        for arg in args:
            self._layers.append(arg)
        for layer in self._layers:
            for param in layer.parameters():
                self._parameters.append(param)

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, item: int) -> type[Layer]:
        return self._layers[item]

    @property
    def layers(self) -> List[type[Layer]]:
        return self._layers

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def __call__(self, x: Tensor):
        return NotImplementedError("forward method must be defined by derived class")

    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()
