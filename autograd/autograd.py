from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import dataclass
from math import exp


def compute_sigmoid(x: int | float):
    return 1 / (1 + exp(-x))


def compute_addition(x: int | float, y: int | float):
    return x + y


def compute_mul(x: int | float, y: int | float):
    return x * y


class Operable:
    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def sigmoid(self):
        return Sigmoid(self)


class Value(Operable):
    def __init__(self, data, grad=0, requires_grad: bool = False):
        self._data = data
        self._grad = grad
        self._requires_grad = requires_grad

    def __str__(self):
        return f"value node -- {self._data}"

    def backward(self, grad):
        self._grad += grad


class UnaryOperation(ABC, Operable):
    def __init__(self, op, grad=0):
        self._op = op
        self._grad = grad
        self._requires_grad = self._op._requires_grad
        if not self._op._requires_grad:
            self.grad_fn = None

    @abstractmethod
    def grad_fn(self):
        pass

    def backward(self, grad=None):
        if not grad:
            grad = 1.0
        self._grad += grad
        if self.grad_fn:
            self._op.backward(self.grad_fn(grad))


class BinaryOperation(ABC, Operable):
    def __init__(self, op1, op2, grad=0):
        self._op1 = op1
        self._op2 = op2
        self._grad = grad
        self._requires_grad = self._op1._requires_grad or self._op2._requires_grad
        if not self._op1._requires_grad:
            self.grad_fn1 = None
        if not self._op2._requires_grad:
            self.grad_fn2 = None

    @abstractmethod
    def grad_fn1(self, grad):
        pass

    @abstractmethod
    def grad_fn2(self, grad):
        pass

    def backward(self, grad=None):
        assert self._requires_grad
        if not grad:
            grad = 1.0
        self._grad += grad
        if self.grad_fn1:
            self._op1.backward(self.grad_fn1(grad))
        if self.grad_fn2:
            self._op2.backward(self.grad_fn2(grad))


class Sigmoid(UnaryOperation):
    def __init__(self, op):
        super().__init__(op)
        self._data = compute_sigmoid(self._op._data)

    def __str__(self):
        return f"sigmoid node -- {self._data}"

    def grad_fn(self, grad):
        return grad * (self._data * (1 - self._data))


class Add(BinaryOperation):
    def __init__(self, op1, op2):
        super().__init__(op1, op2)
        self._data = compute_addition(self._op1._data, self._op2._data)

    def __str__(self):
        return f"add node -- {self._data}"

    def grad_fn1(self, grad):
        return grad

    def grad_fn2(self, grad):
        return grad


class Mul(BinaryOperation):
    def __init__(self, op1, op2):
        super().__init__(op1, op2)
        self._data = compute_mul(self._op1._data, self._op2._data)

    def __str__(self):
        return f"mul node -- {self._data}"

    def grad_fn1(self, grad):
        return grad * self._op2._data

    def grad_fn2(self, grad):
        return grad * self._op1._data


# consider adding children & grad_fns in data structures to remove the 
# need for separate unary, binary ops.

# Child -> struct w/ operand and its grad_fn
# N-ary operation -> list of N dependencies


@dataclass(frozen=True)
class Child:
    operand: Value
    grad_fn: Callable


class NaryOperation(Operable):
    def __init__(self):
        pass