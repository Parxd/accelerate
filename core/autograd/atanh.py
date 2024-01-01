import numpy as np


class Arctanh:
    def __init__(self, x):
        self.x = x
        self.data = np.arctanh(x)

    def __call__(self, grad):
        return (grad / (1 - self.x ** 2),)
