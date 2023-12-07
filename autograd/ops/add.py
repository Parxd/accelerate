from __future__ import annotations

from autograd.shared_types import ValueCtx, Child


def add(left: ValueCtx,
        right: ValueCtx) -> ValueCtx:
    def compute(a: int | float, b: int | float) -> int | float:
        return a + b
    left_grad_fn, right_grad_fn = None, None
    if left.requires_grad:
        def l_grad_fn(grad):
            return grad
        left_grad_fn = l_grad_fn
    if right.requires_grad:
        def r_grad_fn(grad):
            return grad
        right_grad_fn = r_grad_fn
    data = compute(left.data, right.data)
    children = [Child(left, left_grad_fn), Child(right, right_grad_fn)]

    return ValueCtx(data,
                    0,
                    left.requires_grad or right.requires_grad,
                    children)

