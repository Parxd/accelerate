from math import exp
from auto.math.backward_base import BackwardBase
from auto.gradient_context import GradientContext


def sigmoid(a):
    return 1 / (1 + exp(-a))


class SigmoidBackward(BackwardBase):
    def compute_grad(self,
                     context: GradientContext):
        # return as tuple
        return context.parent_grad * (context.parent_data * (1 - context.parent_data)),
