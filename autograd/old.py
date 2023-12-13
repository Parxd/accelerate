from __future__ import annotations
from abc import ABC, abstractmethod
from math import exp

# will likely need to deprecate & refactor this
# the value & un/binary operations classes are essentially the same, making this hard to separate into different
# modules due to circular imports with the operable class


class Operable:
    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def sigmoid(self):
        return Sigmoid(self)


class Value(Operable):
    def __init__(self, data, grad=0, requires_grad: bool = False):
        self._data = data
        self._grad = grad
        self.requires_grad = requires_grad

    def __str__(self):
        return f"value node -- {self._data}"

    # @property attribute in Python allows .data() method to be treated as an attribute
    @property
    def data(self):
        return self._data

    def backward(self, grad):
        self._grad += grad


class UnaryOperation(ABC, Operable):
    def __init__(self, op, data=0, grad=0):
        self._op = op
        self._data = data
        self._grad = grad
        self.requires_grad = self._op.requires_grad
        if not self._op.requires_grad:
            self.grad_fn = None

    @abstractmethod
    def fwd_fn(self, val):
        pass

    @abstractmethod
    def grad_fn(self, grad):
        pass

    @property
    def data(self):
        return self._data

    def backward(self, grad=None):
        if not grad:
            grad = 1.0
        self._grad += grad
        if self.grad_fn:
            self._op.backward(self.grad_fn(grad))


class BinaryOperation(ABC, Operable):
    def __init__(self, op1, op2, data=0, grad=0):
        self._op1 = op1
        self._op2 = op2
        self._data = data
        self._grad = grad
        self.requires_grad = self._op1.requires_grad or self._op2.requires_grad
        if not self._op1.requires_grad:
            self.grad_fn1 = None
        if not self._op2.requires_grad:
            self.grad_fn2 = None

    @abstractmethod
    def fwd_fn(self, x, y):
        pass

    @abstractmethod
    def grad_fn1(self, grad):
        pass

    @abstractmethod
    def grad_fn2(self, grad):
        pass

    @property
    def data(self):
        return self._data

    def backward(self, grad=None):
        assert self.requires_grad
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
        self._data = self.fwd_fn(self._op.data)

    def fwd_fn(self, x):
        return 1 / (1 + exp(-x))

    def __str__(self):
        return f"sigmoid node -- {self._data}"

    def grad_fn(self, grad):
        return grad * (self._data * (1 - self._data))


class Add(BinaryOperation):
    def __init__(self, op1, op2):
        super().__init__(op1, op2)
        self._data = self.fwd_fn(self._op1.data, self._op2.data)

    def __str__(self):
        return f"add node -- {self._data}"

    def fwd_fn(self, x, y):
        return x + y

    def grad_fn1(self, grad):
        return grad

    def grad_fn2(self, grad):
        return grad


class Sub(BinaryOperation):
    def __init__(self, op1, op2):
        super().__init__(op1, op2)
        self._data = self.fwd_fn(self._op1.data, self._op2.data)

    def __str__(self):
        return f"sub node -- {self._data}"

    def fwd_fn(self, x, y):
        return x - y

    def grad_fn1(self, grad):
        return grad

    def grad_fn2(self, grad):
        return -grad


class Mul(BinaryOperation):
    def __init__(self, op1, op2):
        super().__init__(op1, op2)
        self._data = self.fwd_fn(self._op1.data, self._op2.data)

    def __str__(self):
        return f"mul node -- {self._data}"

    def fwd_fn(self, x, y):
        return x * y

    def grad_fn1(self, grad):
        return grad * self._op2.data

    def grad_fn2(self, grad):
        return grad * self._op1.data
