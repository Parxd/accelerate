from auto.math.backward_base import BackwardBase
from auto.gradient_context import GradientContext


def mul(a, b):
    return a * b


class MulBackward(BackwardBase):
    def compute_grad(self,
                     context: GradientContext):
        return (self.left(context.parent_grad,
                          context.op2_data),
                self.right(context.parent_grad,
                           context.op1_data))

    @staticmethod
    def left(grad, r_data):
        return grad * r_data

    @staticmethod
    def right(grad, l_data):
        return grad * l_data
