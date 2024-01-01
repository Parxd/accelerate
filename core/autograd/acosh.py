import numpy as np


class Arccosh:
    def __init__(self, x):
        self.x = x
        self.data = np.arccosh(x)

    def __call__(self, grad):
        return (grad / np.sqrt(self.x ** 2 - 1),)
