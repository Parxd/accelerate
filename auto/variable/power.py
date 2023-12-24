from auto.variable.backward_base import BackwardBase
from auto.gradient_context import GradientContext


def power(a, b):
    return a ** b


class PowBackward(BackwardBase):
    def compute_grad(self,
                     context: GradientContext):
        pass

    @staticmethod
    def _left(grad, r_data):
        pass

    @staticmethod
    def _right(grad, l_data):
        pass
