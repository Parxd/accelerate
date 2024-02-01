from abc import ABC, abstractmethod
from core.tensor import Tensor


class Module(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self._layers = []
        for arg in args:
            self._layers.append(arg)

    @property
    def layers(self):
        return self._layers

    @abstractmethod
    def forward(self, x: Tensor):
        return NotImplementedError("forward method must be defined by derived class")

    def __len__(self):
        return len(self._layers)

    def __str__(self):
        comps: str = ""
        for i in self._layers:
            comps.join(i.__str__())
        return comps
