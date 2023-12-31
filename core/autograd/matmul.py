from .utils import handle_broadcast


class MatMul:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.data = left @ right

    # see /res/matmul_grad for derivation
    # https://math.stackexchange.com/a/3850121
    def __call__(self, grad):
        l_grad = handle_broadcast(self.left, (grad @ self.right.T))
        r_grad = handle_broadcast(self.right, (self.left.T @ grad))
        return l_grad, r_grad
