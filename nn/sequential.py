from typing import List
from .module import Module


class Sequential(Module):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args)

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
