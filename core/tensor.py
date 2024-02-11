from __future__ import annotations
from typing import Callable, Optional, Tuple, Type

from core.operators import *
from core.device import Device

CPU = Device.CPU
GPU = Device.GPU


def np_conv(x):
    # if provided scalar, list, numpy.ndarray
    try:
        return np.asarray(x, dtype=np.float64)
    # else if provided cupy.ndarray
    except TypeError:
        return x.get()


class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None, _children=None, device='cpu'):
        if _children is None:
            _children = ()
        self.data: np.ndarray | cp.ndarray = np_conv(data)
        self.grad: Optional[Tensor] = None
        self.requires_grad: bool = requires_grad
        self.grad_fn: Callable[[np.ndarray | cp.ndarray], Tuple] = grad_fn
        self._children: Tuple[Tensor] = _children
        self.device: Device = CPU if device == 'cpu' else GPU
        self.dtype: np.dtype = self.data.dtype
        self.ndim: int = self.data.ndim
        self.shape: Tuple[int, ...] = self.data.shape
        if self.requires_grad:
            self.grad = Tensor(np.zeros_like(self.data),
                               device='cpu' if self.device == CPU else 'cuda')
        if self.device == GPU:
            self.data = cp.asarray(self.data)

    def __eq__(self, other):
        return np.array_equal(self.data, other.data) and self.dtype == other.dtype

    def __str__(self):
        return (f"Tensor("
                f"{np.array2string(self.data, prefix='Tensor(')}, device={self.device})")

    def __repr__(self):
        return f"{self.data}"

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return self.data.size

    def to(self, new: str):
        new = CPU if new == "cpu" else GPU
        # already on same device
        if self.device == new:
            return
        # gpu -> cpu
        if new == CPU:
            self.data = self.data.get()
            self.device = CPU
        # cpu -> gpu
        if new == GPU:
            self.data = cp.array(self.data)
            self.device = GPU

    def backward(self, grad: Optional[Tensor] = None):
        if self.requires_grad:
            if grad is None:
                if self.ndim != 0:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                else:
                    grad = Tensor(1,
                                  device='cpu' if self.device == CPU else 'cuda')
            self.grad.data += grad.data
            if self.grad_fn:
                gradients = self.grad_fn(self.grad.data)
                for child, gradient in zip(self._children, gradients):
                    child.backward(gradient)

    def detach(self):
        return Tensor(
            self.data,
            requires_grad=False,
            grad_fn=None,
            _children=None,
            device='cpu' if self.device == CPU else 'cuda'
        )

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data),
                           device='cpu' if self.device == CPU else 'cuda')

    def _operation(self,
                   node_type: Type[Node],
                   *args) -> Tensor:
        node = node_type()
        other = args[0] if args else None
        return Tensor(
            data=node.forward(self.data, other.data) if other else node.forward(self.data),
            requires_grad=(self.requires_grad or other.requires_grad) if other else self.requires_grad,
            grad_fn=node.backward,
            _children=(self, other) if other else (self,),
            device='cpu' if self.device == CPU else 'cuda'
        )

    def __add__(self, other) -> Tensor:
        return self._operation(Add, other)

    def __sub__(self, other) -> Tensor:
        return self._operation(Sub, other)

    def __mul__(self, other) -> Tensor:
        return self._operation(Mul, other)

    def exp(self) -> Tensor:
        return self._operation(Exp)

    def sum(self) -> Tensor:
        return self._operation(Sum)

    def mean(self) -> Tensor:
        return self._operation(Mean)

    def sin(self) -> Tensor:
        return self._operation(Sin)
