import numpy as np


class Tanh:
    def __init__(self, x):
        self.x = x
        self.data = np.tanh(x)

    def __call__(self, grad):
        return (grad / np.cosh(self.x) ** 2,)
