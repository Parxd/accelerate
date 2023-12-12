from typing import Callable, Dict, List

from auto.math import add, mul  # namespace error if importing just "math"


grad_fn_mapping: Dict[Callable, List[Callable]] = {
    add: [
        lambda grad: grad,
        lambda grad: grad
    ],
    mul: [
        lambda grad, r_data: grad * r_data,
        lambda grad, l_data: grad * l_data
    ]
}


class Variable:
    def __init__(self, data, grad, requires_grad, children, grad_fn):
        self.data = data
        self.grad = grad
        self.requires_grad = requires_grad
        self.children = children
        self.grad_fn = grad_fn
        # consider adding a bool to check if leaf node or not
        # self.leaf = leaf

    def backward(self, grad=None):
        assert self.requires_grad
        if grad is None:
            grad = 1
        self.grad += grad
        # this is why I don't want to have grad_fn be a list storing multiple gradient functions
        # instead, consider using a class and overriding __call__ like pt
        for grad_fn, child in zip(self.grad_fn, self.children):
            gradient = grad_fn(child)
            child.backward(gradient)

    def __str__(self):
        return \
            f"auto.Variable({self.data}, grad={self.grad}, requires_grad={self.requires_grad}, grad_fn={self.grad_fn}"

    def __add__(self, other):
        data = add(self.data, other.data)
        return Variable(data,
                        0,
                        (self.requires_grad or other.requires_grad),
                        [self, other],
                        grad_fn_mapping[add])

    def __mul__(self, other):
        data = mul(self.data, other.data)
        return Variable(data,
                        0,
                        (self.requires_grad or other.requires_grad),
                        [self, other],
                        grad_fn_mapping[mul])
