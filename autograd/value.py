from __future__ import annotations
from typing import List, Callable
from dataclasses import dataclass

from .ops import *
from .shared_types import ValueCtx, ChildCtx


@dataclass(frozen=True)
class Child:
    """
    Child context class that holds Value context and its corresponding gradient function, assigned by operations
    """
    operand: Value
    grad_fn: Callable


class Value:
    def __init__(self,
                 data: int | float = 0,
                 grad: int | float = 0,
                 requires_grad: bool = False,
                 children: List[Child | ChildCtx] = None,
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
                child.operand.backward(grad)

    def __str__(self) -> str:
        return f'Value({self._data})'

    def __add__(self,
                other: Value) -> Value:
        res = Value.init_context(add(self.context, other.context))
        concrete_children = [Child(self, res.children[0].grad_fn), Child(other, res.children[1].grad_fn)]
        res.children = concrete_children
        return res

    def __sub__(self,
                other: Value) -> Value:
        res = Value.init_context(add(self.context, other.context))
        res.children = [Child(Value.init_context(child.operand_ctx), child.grad_fn) for child in res.children]
        return res

    def __mul__(self,
                other: Value) -> Value:
        res = Value.init_context(add(self.context, other.context))
        res.children = [Child(Value.init_context(child.operand_ctx), child.grad_fn) for child in res.children]
        return res

    def __div__(self,
                other: Value) -> Value:
        res = Value.init_context(add(self.context, other.context))
        res.children = [Child(Value.init_context(child.operand_ctx), child.grad_fn) for child in res.children]
        return res

    def sigmoid(self):
        return Value.init_context(sigmoid(self.context))

    def relu(self):
        return Value.init_context(relu(self.context))
