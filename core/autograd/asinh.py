import numpy as np


class Arcsinh:
    def __init__(self, x):
        self.x = x
        self.data = np.arcsinh(x)

    def __call__(self, grad):
        return (grad / np.sqrt(1 + self.x ** 2),)
