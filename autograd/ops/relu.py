from __future__ import annotations

from autograd.shared_types import ValueCtx, Child


def relu(op: ValueCtx) -> ValueCtx:
    def compute(x: int | float) -> int | float:
        return x * int(x > 0)
    data = compute(op.data)
    grad_fn = None
    if op.requires_grad:
        def grad_func(grad):
            return grad * 0 if op.data < 0 else 1
        grad_fn = grad_func
    children = [Child(op, grad_fn)]
    return ValueCtx(data,
                    0,
                    op.requires_grad,
                    children)
