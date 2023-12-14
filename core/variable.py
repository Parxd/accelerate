from __future__ import annotations
from typing import Callable, List, Type
from auto.math import *
from auto.gradient_context import GradientContext


class Variable:
    def __init__(self,
                 data: int | float,
                 requires_grad: bool = False,
                 grad: int | float = 0,
                 grad_fn: Type[BackwardBase] = None,
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

    def set(self,
            data: int | float):
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
                if child.requires_grad:
                    child.backward(grad_input)

    def __str__(self):
        return \
            f"auto.Variable({self.data}, grad={self.grad}, requires_grad={self.requires_grad}, grad_fn={self.grad_fn}"

    def __neg__(self):
        return self.__mul__(-1)

    def __add__(self, other):
        return self._binary_op(other, add, AddBackward)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, sub, SubBackward)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return self._binary_op(other, mul, MulBackward)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, div, DivBackward)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other, modulo=None):
        return self._binary_op(other, power, PowBackward)

    def sigmoid(self):
        return self._unary_op(sigmoid, SigmoidBackward)

    def relu(self):
        return self._unary_op(relu, ReluBackward)

    def tanh(self):
        return self._unary_op(tanh, TanhBackward)

    def _unary_op(self,
                  op: Callable,
                  grad_fn: Type[BackwardBase]):
        data = op(self.data)
        return Variable(data,
                        self.requires_grad,
                        0,
                        grad_fn() if self.requires_grad else None,
                        [self],
                        False)

    def _binary_op(self,
                   other: int | float | Variable,
                   op: Callable,
                   grad_fn: Type[BackwardBase]):
        if isinstance(other, (int, float)):
            other = Variable(other)
        data = op(self.data, other.data)
        grad_req = self.requires_grad or other.requires_grad
        return Variable(data,
                        grad_req,
                        0,
                        grad_fn() if grad_req else None,
                        [self, other],
                        False)
