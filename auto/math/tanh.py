import math
from auto.math.backward_base import BackwardBase
from auto.gradient_context import GradientContext


def tanh(a):
    return math.tanh(a)


class TanhBackward(BackwardBase):
    def compute_grad(self,
                     context: GradientContext):
        assert context.op2_data is None
        # return as tuple
        return context.parent_grad * (1 - context.parent_data ** 2),
