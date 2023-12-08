from __future__ import annotations
from typing import List

from .ops import *
from .shared_types import ValueCtx, Child


class Value:
    def __init__(self,
                 data: int | float = 0,
                 grad: int | float = 0,
                 requires_grad: bool = False,
                 children: List[Child] = None,
                 context: ValueCtx = None) -> None:
        self._data = data
        self._grad = grad
        self.requires_grad = requires_grad
        self.children = children or []
        self.context = context or ValueCtx(self._data,
                                           self._grad,
                                           self.requires_grad,
                                           self.children)

    @classmethod
    def init_context(cls,
                     ctx: ValueCtx) -> Value:
        return cls(**ctx.__dict__)

    def __str__(self) -> str:
        return f'Value({self._data})'

    def __add__(self,
                other: Value) -> Value:
        return Value.init_context(add(self.context, other.context))

    def __sub__(self,
                other: Value) -> Value:
        return Value.init_context(sub(self.context, other.context))

    def __rsub__(self,
                 other: Value) -> Value:
        return Value.init_context(sub(other.context, self.context))

    def __mul__(self,
                other: Value) -> Value:
        return Value.init_context(mul(self.context, other.context))

    def backward(self,
                 grad: int | float = None) -> None:
        pass
