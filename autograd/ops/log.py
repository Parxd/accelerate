from __future__ import annotations
import math

from ..shared_types import ValueCtx


def log(op: ValueCtx) -> ValueCtx:
    def compute(x: int | float) -> int | float:
        return math.log(x)
    data = compute(op.data)
    grad_fn = None
    if op.requires_grad:
        def grad_func(grad):
            return grad / op.data
        grad_fn = grad_func
    children = grad_fn
    return ValueCtx(data,
                    0,
                    op.requires_grad,
                    children)
