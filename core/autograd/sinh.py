import numpy as np


class Sinh:
    def __init__(self, x):
        self.x = x
        self.data = np.sinh(x)

    def __call__(self, grad):
        return (grad * np.cosh(self.x),)
