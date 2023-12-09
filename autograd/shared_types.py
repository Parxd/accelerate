from __future__ import annotations
from typing import List, Callable
from dataclasses import dataclass


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
    children: List[Callable]  # misleading name; it's the children's grad functions, not children themselves
