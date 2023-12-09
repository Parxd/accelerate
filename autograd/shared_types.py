from __future__ import annotations
from typing import List, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ChildCtx:
    """
    Child context class that holds Value context and its corresponding gradient function, assigned by operations
    """
    operand_ctx: ValueCtx
    grad_fn: Callable


@dataclass
class ValueCtx:
    """
    Context class for Value, exposing bare minimum for operations to act on the Value type without knowing about
    Value itself to prevent circular import issues

    Should never be used independently, but instead only wrapped by Value
    """
    data: int | float
    grad: int | float
    requires_grad: bool
    children: List[ChildCtx]
