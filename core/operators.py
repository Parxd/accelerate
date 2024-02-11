from core.node import *


class Exp(Node):
    def __init__(self):
        self._data = None
        self._saved_result: np.ndarray | cp.ndarray = None

    def forward(self,
                x: np.ndarray | cp.ndarray):
        self._saved_result = np.exp(x)
        return self._saved_result

    def backward(self, gradient):
        return gradient * self._saved_result,


class Mul(Node):
    def __init__(self):
        self._lhs, self._rhs = None, None
        self._saved_result: np.ndarray | cp.ndarray = None

    def forward(self,
                x: np.ndarray | cp.ndarray,
                y: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        self._lhs, self._rhs = x, y
        return x * y

    def backward(self, gradient: np.ndarray | cp.ndarray):
        return gradient * self._rhs, gradient * self._lhs
