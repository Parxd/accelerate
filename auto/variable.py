from auto.math import *
from .gradient_context import GradientContext


class Variable:
    def __init__(self, data, requires_grad, grad=0, grad_fn=None, children=None, leaf=True):
        if children is None:
            children = []
        self.data = data
        self.requires_grad = requires_grad
        self.grad = grad
        self.grad_fn = grad_fn
        self.children = children
        # check if leaf node in graph
        self.leaf = leaf

    def backward(self, grad=None):
        assert self.requires_grad
        if grad is None:
            grad = 1
        self.grad += grad
        if not self.leaf:
            gradients = self.grad_fn.compute_grad(GradientContext(
                self.children[0].data,
                self.children[1].data,
                self.data,
                self.grad
            ))
            self.children[0].backward(gradients[0])
            self.children[1].backward(gradients[1])

    def __str__(self):
        return \
            f"auto.Variable({self.data}, grad={self.grad}, requires_grad={self.requires_grad}, grad_fn={self.grad_fn}"

    def __add__(self, other):
        data = add(self.data, other.data)
        return Variable(data,
                        self.requires_grad or other.requires_grad,
                        0,
                        AddBackward(),
                        [self, other],
                        False)

    def __mul__(self, other):
        data = mul(self.data, other.data)
        return Variable(data,
                        self.requires_grad or other.requires_grad,
                        0,
                        MulBackward(),
                        [self, other],
                        False)

    def sigmoid(self):
        data = sigmoid(self.data)
        return Variable(data,
                        self.requires_grad,
                        0,
                        SigmoidBackward(),
                        [self],
                        False)
