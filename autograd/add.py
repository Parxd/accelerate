from .shared_types import ValueCtx, Child


def add(a: ValueCtx, b: ValueCtx) -> ValueCtx:
    left_grad_fn, right_grad_fn = None, None
    if a.requires_grad:
        def l_grad_fn(grad):
            return grad
        left_grad_fn = l_grad_fn
    if b.requires_grad:
        def r_grad_fn(grad):
            return grad
        right_grad_fn = r_grad_fn
    comp = a.data + b.data
    children = [Child(a, left_grad_fn), Child(b, right_grad_fn)]

    return ValueCtx(comp, 0, a.requires_grad or b.requires_grad, children)

