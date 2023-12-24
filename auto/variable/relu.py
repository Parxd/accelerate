from auto.variable.backward_base import BackwardBase
from auto.gradient_context import GradientContext


def relu(a):
    return int(a > 0) * a


class ReluBackward(BackwardBase):
    def compute_grad(self,
                     context: GradientContext):
        assert context.op2_data is None
        # return as tuple
        return context.parent_grad * int(context.op1_data > 0) * 1,
