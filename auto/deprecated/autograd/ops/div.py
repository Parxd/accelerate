from __future__ import annotations

from ..shared_types import ValueCtx


def div(left: ValueCtx,
        right: ValueCtx) -> ValueCtx:
    def compute(a: int | float,
                b: int | float) -> int | float:
        return a / b
    data = compute(left.data, right.data)
    left_grad_fn, right_grad_fn = None, None
    if left.requires_grad:
        def l_grad_fn(grad):
            return grad / right.data
        left_grad_fn = l_grad_fn
    if right.requires_grad:
        def r_grad_fn(grad):
            return grad * (-left.data / (right.data ** 2))
        right_grad_fn = r_grad_fn
    children = [left_grad_fn, right_grad_fn]
    return ValueCtx(data,
                    0,
                    left.requires_grad or right.requires_grad,
                    children)
