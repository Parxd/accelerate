from __future__ import annotations
from typing import List

from .ops import *
from .shared_types import ValueCtx, Child


class Value:
    def __init__(self,
                 data: int | float,
                 grad: int | float = 0,
                 requires_grad: bool = False,
                 children: List[Child] = None):
        if children is None:
            children = []
        self._data = data
        self._grad = grad
        self.requires_grad = requires_grad
        self.children = children
        self.context = ValueCtx(self._data,
                                self._grad,
                                self.requires_grad,
                                self.children)

    def __str__(self) -> str:
        return f'Value({self._data})'

    def __add__(self,
                other: Value) -> Value:
        return Value(add(self.context, other.context))

    def __mul__(self,
                other: Value) -> Value:
        return Value(mul(self.context, other.context))

    def backward(self,
                 grad: int | float = None) -> int | float | Value:
        pass
