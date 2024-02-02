from typing import List
from .module import Module


class Sequential(Module):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args)

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, item: int):
        return self._children[item]

    def __iter__(self):
        for child in self._children:
            yield child

    def __str__(self):
        lines = [f"{self.__class__.__name__}("]
        for i, module in enumerate(self):
            lines.append(f"\t({i}): {module}")
        lines.append(")")
        print('\n'.join(lines))

    def forward(self, x):
        for layer in self._children:
            x = layer(x)
        return x
