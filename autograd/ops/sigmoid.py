from __future__ import annotations
from math import exp

from autograd.shared_types import ValueCtx, ChildCtx


def sigmoid(op: ValueCtx) -> ValueCtx:
    def compute(x: int | float) -> int | float:
        return 1 / (1 + exp(-x))
    data = compute(op.data)
    grad_fn = None
    if op.requires_grad:
        def grad_func(grad):
            return grad * (data * (1.0 - data))
        grad_fn = grad_func
    children = [ChildCtx(op, grad_fn)]
    return ValueCtx(data,
                    0,
                    op.requires_grad,
                    children)
