import numpy as np


class Arctan:
    def __init__(self, x):
        self.x = x
        self.data = np.arctan(x)

    def __call__(self, grad):
        return (grad / (1 + self.x ** 2),)
