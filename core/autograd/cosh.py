import numpy as np


class Cosh:
    def __init__(self, x):
        self.x = x
        self.data = np.cosh(x)

    def __call__(self, grad):
        return (grad * np.sinh(self.x),)
