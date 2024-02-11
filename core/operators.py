from core.node import *


class Exp(Node):
    def __init__(self):
        self._saved_result: np.ndarray | cp.ndarray = None

    def forward(self,
                x: np.ndarray | cp.ndarray):
        self._saved_result = np.exp(x)
        return self._saved_result

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient * self._saved_result,


class Sum(Node):
    def __init__(self):
        ...

    def forward(self,
                x: np.ndarray | cp.ndarray):
        return x.sum()

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient,


class Mean(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: np.ndarray | cp.ndarray):
        self._data = x
        return x.mean()

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient / self._data.size,


class Sin(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: np.ndarray | cp.ndarray):
        self._data = x
        return np.sin(x)

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient * np.cos(self._data),


class Cos(Node):
    def __init__(self):
        self._data = None

    def forward(self,
                x: np.ndarray | cp.ndarray):
        self._data = x
        return np.cos(x)

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient * -np.sin(self._data),


class Add(Node):
    def __init__(self):
        ...

    def forward(self,
                x: np.ndarray | cp.ndarray,
                y: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return x + y

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient, gradient


class Sub(Node):
    def __init__(self):
        ...

    def forward(self,
                x: np.ndarray | cp.ndarray,
                y: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        return x - y

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient, -gradient


class Mul(Node):
    def __init__(self):
        self._lhs, self._rhs = None, None

    def forward(self,
                x: np.ndarray | cp.ndarray,
                y: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        self._lhs, self._rhs = x, y
        return x * y

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient * self._rhs, gradient * self._lhs


