from __future__ import annotations
from typing import List, Callable
from dataclasses import dataclass

from .ops import *
from .shared_types import ValueCtx


@dataclass(frozen=True)
class Child:
    operand: Value = None
    grad_fn: Callable = None


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

    def backward(self,
                 grad: int | float = None) -> None:
        assert self.requires_grad
        if not grad:
            grad = 1
        self._grad += grad
        for child in self.children:
            if child.grad_fn:
                child.operand.backward(child.grad_fn(grad))

    def __str__(self) -> str:
        return f'Value({self._data})'

    def __add__(self,
                other: Value) -> Value:
        res_ctx = add(self.context, other.context)
        res = Value.init_context(res_ctx)
        res.children = [Child(self, res_ctx.children[0]), Child(other, res_ctx.children[1])]
        return res

    def __sub__(self,
                other: Value) -> Value:
        pass

    def __mul__(self,
                other: Value) -> Value:
        # res_ctx = mul(self.context, other.context)
        # children = [Child(self, res_ctx.children[0]), Child(other, res_ctx.children[1])]
        # res = Value.init_context(res_ctx)
        # res.children = children
        # return res
        pass

    def __div__(self,
                other: Value) -> Value:
        pass

    def sigmoid(self) -> Value:
        pass

    def relu(self) -> Value:
        pass
