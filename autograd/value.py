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
                 children: List[Child | Callable] = None,
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

    def _wrap_children(self,
                       left_value: Value,
                       right_value: Value,
                       grad_fn_list: List[Callable]) -> Value:
        pass

    def __str__(self) -> str:
        return f'Value({self._data})'

    def __add__(self,
                other: Value) -> Value:
        res = Value.init_context(add(self.context, other.context))
        # after this line, res.children is just a list of raw gradient functions--the one directly from the
        # ValueContext object
        # we need to wrap the raw gradient function with its corresponding Value object
        # i.e. encapsulate the left grad_fn with self, and the right grad_fn with other
        # this is why Value.children is annotated as List[Child | Callable]; at one point of running, it will have to
        # hold onto the raw gradient callables List[Callable] until the next two lines encapsulate them and
        # reassign them to a List[Child]
        wrapped_children = [Child(obj, raw_grad_fn) for obj, raw_grad_fn in zip([self, other], res.children)]
        res.children = wrapped_children
        return res

    def __sub__(self,
                other: Value) -> Value:
        res = Value.init_context(sub(self.context, other.context))
        wrapped_children = [Child(obj, raw_grad_fn) for obj, raw_grad_fn in zip([self, other], res.children)]
        res.children = wrapped_children
        return res

    def __mul__(self,
                other: Value) -> Value:
        res = Value.init_context(mul(self.context, other.context))
        wrapped_children = [Child(obj, raw_grad_fn) for obj, raw_grad_fn in zip([self, other], res.children)]
        res.children = wrapped_children
        return res

    def __div__(self,
                other: Value) -> Value:
        res = Value.init_context(div(self.context, other.context))
        wrapped_children = [Child(obj, raw_grad_fn) for obj, raw_grad_fn in zip([self, other], res.children)]
        res.children = wrapped_children
        return res

    def sigmoid(self) -> Value:
        res = Value.init_context(sigmoid(self.context))
        wrapped_children = Child(self, res.children[0])
        res.children = wrapped_children
        return res

    def relu(self) -> Value:
        res = Value.init_context(relu(self.context))
        wrapped_children = Child(self, res.children[0])
        res.children = wrapped_children
        return res
