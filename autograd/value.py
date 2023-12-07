from __future__ import annotations
from typing import List

from .shared_types import ValueCtx, Child
from .add import add


class Value(ValueCtx):
    def __init__(self,
                 data: int | float,
                 grad: int | float = 0,
                 requires_grad: bool = False,
                 children: List[Child] = None):
        super().__init__(data, grad, requires_grad, children)

    def __str__(self):
        return f'Value({self.data})'

    def __add__(self, other):
        res = add(self, other)  # ValueType
        return Value(res.data, res.grad, res.requires_grad, res.children)
