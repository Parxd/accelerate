from core.node import *


Array = np.ndarray | cp.ndarray


def handle_broadcast(tensor: Array,
                     grad: Array) -> Array:
    """
    Handles gradient summing when broadcasting np.ndarray
    Use with all binary operations that support broadcasting
    """
    dims_added = grad.ndim - tensor.ndim
    for _ in range(dims_added):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(tensor.shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Neg(Node):
    def __init__(self):
        ...

    def forward(self,
                x: Array) -> Array:
        return -x

    def backward(self, gradient: Array) -> Array:
        return -gradient,


class Exp(Node):
    def __init__(self):
        self._saved_result: Array = None

    def forward(self,
                x: Array) -> Array:
        self._saved_result = np.exp(x)
        return self._saved_result

    def backward(self, gradient: Array) -> Array:
        return gradient * self._saved_result,


class Log(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: Array) -> Array:
        self._data = x
        return x.log()

    def backward(self, gradient: Array) -> Array:
        return gradient / self._data,


class Sigmoid(Node):
    def __init__(self):
        self._saved_result = None

    def forward(self,
                x: Array) -> Array:
        self._saved_result = 1 / (1 + np.exp(-x))
        return self._saved_result

    def backward(self, gradient: Array) -> Array:
        return gradient * (self._saved_result * (1 - self._saved_result))


class ReLU(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: Array) -> Array :
        self._data = x
        return x * bool(x > 0)

    def backward(self, gradient: Array) -> Array:
        return gradient * (bool(self._data > 0) * 1),


class Square(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: Array) -> Array:
        self._data = x
        return x ** 2

    def backward(self, gradient: Array) -> Array:
        return gradient * 2 * self._data


class Sqrt(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: Array) -> Array:
        return np.sqrt(x)

    def backward(self, gradient: Array) -> Array:
        return gradient * (1 / 2) * self._data ** (-1 / 2),


class Sum(Node):
    def __init__(self):
        ...

    def forward(self,
                x: Array) -> Array:
        return x.sum()

    def backward(self, gradient: Array) -> Array:
        return gradient,


class Mean(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: Array) -> Array:
        self._data = x
        return x.mean()

    def backward(self, gradient: Array) -> Array:
        return gradient / self._data.size,


class Transpose(Node):
    def __init__(self):
        ...

    def forward(self,
                x: Array) -> Array:
        return x.T

    def backward(self, gradient: Array) -> Array:
        return gradient.T,


class Sin(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: Array) -> Array:
        self._data = x
        return np.sin(x)

    def backward(self, gradient: Array) -> Array:
        return gradient * np.cos(self._data),


class Cos(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: Array) -> Array:
        self._data = x
        return np.cos(x)

    def backward(self, gradient: Array) -> Array:
        return gradient * -np.sin(self._data),


class Tan(Node):
    def __init__(self):
        self._data = None

    def forward(self, x: Array) -> Array:
        self._data = x
        return np.tan(x)

    def backward(self, gradient: Array) -> Array:
        return gradient / np.cos(self._data) ** 2


class Add(Node):
    def __init__(self):
        self._lhs, self._rhs = None, None

    def forward(self,
                x: Array,
                y: Array) -> Array:
        self._lhs, self._rhs = x, y
        return x + y

    def backward(self, gradient: Array) -> Array:
        return (
            handle_broadcast(self._lhs, gradient),
            handle_broadcast(self._rhs, gradient)
        )


class Sub(Node):
    def __init__(self):
        self._lhs, self._rhs = None, None

    def forward(self,
                x: Array,
                y: Array) -> Array:
        self._lhs, self._rhs = x, y
        return x - y

    def backward(self, gradient: Array) -> Array:
        return (
            handle_broadcast(self._lhs, gradient),
            handle_broadcast(self._rhs, -gradient)
        )


class Mul(Node):
    def __init__(self):
        self._lhs, self._rhs = None, None

    def forward(self,
                x: Array,
                y: Array) -> Array:
        self._lhs, self._rhs = x, y
        return x * y

    def backward(self, gradient: Array) -> Array:
        return (
            handle_broadcast(self._lhs, (gradient * self._rhs)),
            handle_broadcast(self._rhs, (gradient * self._lhs))
        )


class Div(Node):
    def __init__(self):
        self._lhs, self._rhs = None, None

    def forward(self,
                x: Array,
                y: Array) -> Array:
        self._lhs, self._rhs = x, y
        return x / y

    def backward(self, gradient: Array) -> Array:
        return (
            handle_broadcast(self._lhs, (gradient / self._rhs)),
            handle_broadcast(self._rhs, (gradient * (-self._lhs / self._rhs ** 2)))
        )


class MatMul(Node):
    def __init__(self):
        self._lhs, self._rhs = None, None

    def forward(self,
                x: Array,
                y: Array) -> Array:
        self._lhs, self._rhs = x, y
        return x @ y

    def backward(self, gradient: Array) -> Array:
        return (
            handle_broadcast(self._lhs, (gradient @ self._rhs.T)),
            handle_broadcast(self._rhs, (self._lhs.T @ gradient))
        )
