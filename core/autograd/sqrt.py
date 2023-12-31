import numpy as np


class Sqrt:
    def __init__(self, x):
        self.x = x
        self.data = np.sqrt(x)

    def __call__(self, grad):
        return (grad * 0.5 * self.x ** -0.5,)
