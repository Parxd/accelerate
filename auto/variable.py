from __future__ import annotations
from typing import Callable, List
from auto.math import *
from auto.gradient_context import GradientContext


class Variable:
    def __init__(self,
                 data: int | float,
                 requires_grad: bool = False,
                 grad: int | float = 0,
                 grad_fn: BackwardBase = None,
                 children: List[Variable] = None,
                 leaf: bool = True):
        if children is None:
            children = []
        self.data = data
        self.requires_grad = requires_grad
        self.grad = grad
        self.grad_fn = grad_fn
        self.children = children
        # check if leaf node in graph
        self.leaf = leaf

    def set(self, data):
        self.data = data
        self.clear_grad()

    def clear_grad(self):
        self.grad = 0

    def backward(self,
                 grad: int | float = None):
        assert self.requires_grad
        if grad is None:
            grad = 1
        self.grad += grad
        if not self.leaf:
            child_data = [child.data if child else None for child in self.children]
            child_data.append(None) if len(child_data) == 1 else child_data
            gradients = self.grad_fn.compute_grad(
                GradientContext(
                    *child_data,
                    self.data,
                    self.grad
                )
            )
            for child, grad_input in zip(self.children, gradients):
                child.backward(grad_input)

    def __str__(self):
        return \
            f"auto.Variable({self.data}, grad={self.grad}, requires_grad={self.requires_grad}, grad_fn={self.grad_fn}"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            data = add(self.data, other)
            grad_req = self.requires_grad
            children = [self]
        else:
            data = add(self.data, other.data)
            grad_req = self.requires_grad or other.requires_grad
            children = [self, other]
        return Variable(data,
                        grad_req,
                        0,
                        AddBackward() if grad_req else None,
                        children,
                        False)

    def __sub__(self, other):
        data = sub(self.data, other.data)
        grad_req = self.requires_grad or other.requires_grad
        return Variable(data,
                        grad_req,
                        0,
                        SubBackward() if grad_req else None,
                        [self, other],
                        False)

    def __mul__(self, other):
        data = mul(self.data, other.data)
        grad_req = self.requires_grad or other.requires_grad
        return Variable(data,
                        grad_req,
                        0,
                        MulBackward() if grad_req else None,
                        [self, other],
                        False)

    def sigmoid(self):
        data = sigmoid(self.data)
        return Variable(data,
                        self.requires_grad,
                        0,
                        SigmoidBackward() if self.requires_grad else None,
                        [self],
                        False)

    def relu(self):
        data = relu(self.data)
        return Variable(data,
                        self.requires_grad,
                        0,
                        ReluBackward() if self.requires_grad else None,
                        [self],
                        False)

    def tanh(self):
        data = tanh(self.data)
        return Variable(data,
                        self.requires_grad,
                        0,
                        TanhBackward() if self.requires_grad else None,
                        [self],
                        False)
