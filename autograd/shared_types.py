from __future__ import annotations
from typing import List, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Child:
    """
    Child class that holds Value context and its corresponding gradient function, assigned by operations
    """
    operand: ValueCtx
    grad_fn: Callable


class ValueCtx:
    """
    Context class for Value, exposing the bare minimum for operations to act on the Value type
    without knowing about Value itself to prevent circular import issues

    Should not be used on its own, but instead only by Value
    """
    def __init__(self,
                 data: int | float,
                 grad: int | float = 0,
                 requires_grad: bool = False,
                 children: List[Child] = None):
        if children is None:
            children = []
        self.data = data
        self.grad = grad
        self.requires_grad = requires_grad
        self.children = children
