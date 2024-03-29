from __future__ import annotations
import warnings
from typing import Callable, Optional, Tuple, Type
from core.operators import *
from core.device import Device

CPU = Device.CPU
GPU = Device.GPU


def tensor_conv(x: int | float | list | np.ndarray | cp.ndarray | Tensor,
                device: Device) -> Tensor:
    if hasattr(x, "grad"):
        return x
    return Tensor(x,
                  device='cpu' if device == CPU else 'gpu')


class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None, _children=None, device='cpu') -> None:
        if _children is None:
            _children = ()
        self.data: np.ndarray | cp.ndarray = np.asarray(data, dtype=np.float64) if device == 'cpu' else cp.asarray(data, dtype=cp.float64)
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
                               device='cpu' if self.device is CPU else 'cuda')

    def __eq__(self, other) -> bool:
        return (np.allclose(self.data, other.data) and
                np.allclose(self.grad.data, other.grad.data) if self.grad else ... and
                self.device is other.device)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (f"Tensor("
                f"{np.array2string(self.data, prefix='Tensor(')}, requires_grad={self.requires_grad}, "
                f"device={self.device})")

    def __repr__(self):
        return f"{self.data}"

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return self.data.size

    def to(self, new: str) -> None:
        new = CPU if new == "cpu" else GPU
        # already on same device
        if self.device == new:
            return
        # gpu -> cpu
        if new == CPU:
            self.data = self.data.get()
            if self.requires_grad:
                self.grad.to("cpu")
            self.device = CPU
        # cpu -> gpu
        if new == GPU:
            self.data = cp.asarray(self.data)
            if self.requires_grad:
                self.grad.to("cuda")
            self.device = GPU

    def backward(self, grad: Optional[Tensor] = None) -> None:
        if self.requires_grad:
            if grad is None:
                if self.ndim != 0:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                else:
                    grad = Tensor(1,
                                  device='cpu' if self.device is CPU else 'cuda')
            self.grad.data += grad.data
            if self.grad_fn:
                gradients = self.grad_fn(self.grad.data)
                for child, gradient in zip(self._children, gradients):
                    # re-wrap incoming gradient array as Tensor
                    child.backward(Tensor(gradient,
                                          device='cpu' if self.device is CPU else 'cuda'))

    # shallow mem-copy disconnected from autograd graph
    def detach(self) -> Tensor:
        return Tensor(
            self.data,
            requires_grad=False,
            grad_fn=None,
            _children=None,
            device='cpu' if self.device is CPU else 'cuda'
        )

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data),
                           device='cpu' if self.device is CPU else 'cuda')

    def numpy(self) -> np.ndarray:
        return np.asarray(self.data)

    def _operation(self,
                   node_type: Type[Node],
                   reverse: bool = False,
                   *args) -> Tensor:
        node = node_type()
        other = tensor_conv(args[0], self.device) if args else None
        if not other:
            data = node.forward(self.data)
        else:
            data = node.forward(other.data, self.data) if reverse else node.forward(self.data, other.data)
        return Tensor(
            data=data,
            requires_grad=(self.requires_grad or other.requires_grad) if other else self.requires_grad,
            grad_fn=node.backward,
            _children=(self, other) if other else (self,),
            device='cpu' if not hasattr(data, "get") else 'cuda'
        )

    def __neg__(self) -> Tensor:
        return self._operation(Neg)

    def __add__(self, other) -> Tensor:
        return self._operation(Add, False, other)

    def __radd__(self, other) -> Tensor:
        return self._operation(Add, True, other)

    def __iadd__(self, other) -> Tensor:
        self.data += other.data
        warnings.warn("in-place operations may break autograd connections")

    def __sub__(self, other) -> Tensor:
        return self._operation(Sub, False, other)

    def __rsub__(self, other) -> Tensor:
        return self._operation(Sub, True, other)

    def __isub__(self, other):
        self.data -= other.data
        warnings.warn("in-place operations may break autograd connections")

    def __mul__(self, other) -> Tensor:
        return self._operation(Mul, False, other)

    def __rmul__(self, other) -> Tensor:
        return self._operation(Mul, True, other)

    def __truediv__(self, other) -> Tensor:
        return self._operation(Div, False, other)

    def __rtruediv__(self, other) -> Tensor:
        return self._operation(Div, True, other)

    def __matmul__(self, other) -> Tensor:
        return self._operation(MatMul, False, other)

    # deep mem-copy still connected to autograd graph
    def clone(self) -> Tensor:
        return self._operation(Clone)

    def transpose(self) -> Tensor:
        return self._operation(Transpose)

    def exp(self) -> Tensor:
        return self._operation(Exp)

    def log(self) -> Tensor:
        return self._operation(Log)

    def sigmoid(self) -> Tensor:
        return self._operation(Sigmoid)

    def relu(self) -> Tensor:
        return self._operation(ReLU)

    def square(self) -> Tensor:
        return self._operation(Square)

    def sqrt(self) -> Tensor:
        return self._operation(Sqrt)

    def sum(self) -> Tensor:
        return self._operation(Sum)

    def mean(self) -> Tensor:
        return self._operation(Mean)

    def sin(self) -> Tensor:
        return self._operation(Sin)

    def cos(self) -> Tensor:
        return self._operation(Cos)

    def tan(self) -> Tensor:
        return self._operation(Tan)
