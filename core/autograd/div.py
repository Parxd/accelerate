from .utils import handle_broadcast


class Div:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.data = left / right

    def __call__(self, grad):
        l_grad = handle_broadcast(self.left, (grad / self.right))
        r_grad = handle_broadcast(self.right, (-grad * self.left / self.right ** 2))
        return l_grad, r_grad
