from auto.math.backward_base import BackwardBase
from auto.gradient_context import GradientContext


def sub(a, b):
    return a - b


class SubBackward(BackwardBase):
    def compute_grad(self,
                     context: GradientContext):
        return self._left(context.parent_grad), self._right(context.parent_grad)

    @staticmethod
    def _left(grad):
        return grad

    @staticmethod
    def _right(grad):
        return -grad
