from abc import ABC, abstractmethod
from auto.gradient_context import GradientContext


class BackwardBase(ABC):
    @abstractmethod
    def compute_grad(self,
                     context: GradientContext):
        raise NotImplementedError
