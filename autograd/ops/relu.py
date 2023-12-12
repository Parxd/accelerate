from __future__ import annotations

from ..shared_types import ValueCtx


def relu(op: ValueCtx) -> ValueCtx:
    def compute(x: int | float) -> int | float:
        return x * int(x > 0)
    data = compute(op.data)
    grad_fn = None
    if op.requires_grad:
        def grad_func(grad):
            temp = int(op.data > 0)
            return int(op.data > 0) * 1 * grad
        grad_fn = grad_func
    children = grad_fn
    return ValueCtx(data,
                    0,
                    op.requires_grad,
                    children)
