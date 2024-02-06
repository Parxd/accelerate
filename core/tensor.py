from .tensorimpl import TensorCPUBackend, TensorGPUBackend


class Tensor:
    def __init__(self,
                 data,
                 requires_grad=False,
                 grad_fn=None,
                 children=None,
                 leaf=True,
                 device='cpu') -> None:
        backend_cls = TensorCPUBackend if device == 'cpu' else TensorGPUBackend
        self._backend = backend_cls(data,
                                    requires_grad,
                                    grad_fn,
                                    children,
                                    leaf)
        self.data = self._backend.data
        self.grad = self._backend.grad
        self.requires_grad = self._backend.requires_grad
        self.grad_fn = self._backend.grad_fn
        self.children = self._backend.children
        self.leaf = self._backend.leaf
        self.device = device

    @classmethod
    def random(cls, shape, requires_grad, device):
        return TensorCPUBackend.random(shape, requires_grad) if device == 'cpu'\
            else TensorGPUBackend.random(shape, requires_grad)

    def zero_grad(self):
        self._backend.zero_grad()

    def backward(self, grad):
        self._backend.backward(grad)

    def transpose(self):
        return self._backend.transpose()

    def sum(self):
        return self._backend.sum()

    def __neg__(self):
        return self._backend.__neg__()

    def exp(self):
        return self._backend.exp()

    def log(self):
        return self._backend.log()

    def square(self):
        return self._backend.square()

    def sqrt(self):
        return self._backend.sqrt()

    def mean(self):
        return self._backend.mean()

    def sin(self):
        return self._backend.sin()

    def cos(self):
        return self._backend.cos()

    def tan(self):
        return self._backend.tan()

    def arcsin(self):
        return self._backend.arcsin()

    def arccos(self):
        return self._backend.arccos()

    def arctan(self):
        return self._backend.arctan()

    def sinh(self):
        return self._backend.sinh()

    def cosh(self):
        return self._backend.cosh()

    def tanh(self):
        return self._backend.tanh()

    def arcsinh(self):
        return self._backend.arcsinh()

    def arccosh(self):
        return self._backend.arccosh()

    def arctanh(self):
        return self._backend.arctanh()

    def relu(self):
        return self._backend.relu()

    def __add__(self, other):
        return self._backend.__add__(other.backend)

    def __radd__(self, other):
        return self._backend.__radd__(other.backend)

    def __sub__(self, other):
        return self._backend.__sub__(other.backend)

    def __rsub__(self, other):
        return self._backend.__rsub__(other.backend)

    def __mul__(self, other):
        return self._backend.__mul__(other.backend)

    def __rmul__(self, other):
        return self._backend.__rmul__(other.backend)

    def __truediv__(self, other):
        return self._backend.__truediv__(other.backend)

    def __matmul__(self, other):
        return self._backend.__matmul__(other.backend)

    # TODO: Fix these
    def __iadd__(self, other):
        ...

    def __isub__(self, other):
        ...

    def __imul__(self, other):
        ...

    def abs(self):
        return self._backend.abs()

    def sigmoid(self):
        return self._backend.sigmoid()

    @property
    def backend(self):
        """
        Returns the backend of the Tensor interface
        Not to be used directly, use provided methods of Tensor
        :return: TensorCPUBackend if device == 'cpu', else TensorGPUBackend
        """
        return self._backend
