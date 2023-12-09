from __future__ import annotations
from typing import List, Callable
from dataclasses import dataclass

from .ops import *
from .shared_types import ValueCtx


@dataclass(frozen=True)
class Child:
    operand: Value = None
    grad_fn: Callable = None


def _wrap_child_unary(value: Value,
                      grad_fn: Callable) -> List[Child]:
    return [Child(value, grad_fn)]


def _wrap_children_binary(left_value: Value,
                          right_value: Value,
                          grad_fn_list: List[Callable]) -> List[Child]:
    return [Child(value, grad_fn) for value, grad_fn in zip([left_value, right_value], grad_fn_list)]


class Value:
    def __init__(self,
                 data: int | float = 0,
                 grad: int | float = 0,
                 requires_grad: bool = False,
                 children: List[Child] = None,
                 children_fns: Callable | List[Callable] = None,
                 context: ValueCtx = None) -> None:
        self._data = data
        self._grad = grad
        self.requires_grad = requires_grad
        self.children = children or []
        self.children_fns = children_fns or []
        self.context = context or ValueCtx(self._data,
                                           self._grad,
                                           self.requires_grad,
                                           self.children_fns)

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
        res = Value.init_context(add(self.context, other.context))
        # after this line, res.children_fns is just a list of raw gradient functions--the one directly from the
        # ValueContext object
        # we need to wrap the raw gradient function with its corresponding Value object and assign to res.children
        # i.e. encapsulate the left grad_fn with self, and the right grad_fn with other for binary operations
        res.children = _wrap_children_binary(self, other, res.children_fns)
        return res

    def __sub__(self,
                other: Value) -> Value:
        res = Value.init_context(sub(self.context, other.context))
        res.children = _wrap_children_binary(self, other, res.children_fns)
        return res

    def __mul__(self,
                other: Value) -> Value:
        res = Value.init_context(mul(self.context, other.context))
        res.children = _wrap_children_binary(self, other, res.children_fns)
        return res

    def __div__(self,
                other: Value) -> Value:
        res = Value.init_context(div(self.context, other.context))
        res.children = _wrap_children_binary(self, other, res.children_fns)
        return res

    def log(self) -> Value:
        res = Value.init_context(log(self.context))
        res.children = _wrap_child_unary(self, res.children_fns)
        return res

    def exp(self) -> Value:
        res = Value.init_context(exp(self.context))
        res.children = _wrap_child_unary(self, res.children_fns)
        return res

    def sigmoid(self) -> Value:
        res = Value.init_context(sigmoid(self.context))
        res.children = _wrap_child_unary(self, res.children_fns)
        return res

    def relu(self) -> Value:
        res = Value.init_context(relu(self.context))
        res.children = _wrap_child_unary(self, res.children_fns)
        return res

    def sin(self) -> Value:
        res = Value.init_context(sin(self.context))
        res.children = _wrap_child_unary(self, res.children_fns)
        return res

    def cos(self) -> Value:
        res = Value.init_context(cos(self.context))
        res.children = _wrap_child_unary(self, res.children_fns)
        return res
