from .module import Module


class Sequential(Module):
    # args is a tuple: ex. (Linear, Sigmoid, Linear)
    def __init__(self,
                 *args) -> None:
        super().__init__(args)
        self._index = 0

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, item: int):
        return self._children[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._children):
            layer = self._children[self._index]
            self._index += 1
            return layer
        else:
            raise StopIteration

    def __str__(self):
        lines = [f"{self.__class__.__name__}("]
        for i, module in enumerate(self):
            lines.append(f"\t({i}): {module}")
        lines.append(")")
        return '\n'.join(lines)

    def forward(self, x):
        for layer in self._children:
            x = layer(x)
        return x
