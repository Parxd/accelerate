from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        return []

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError
