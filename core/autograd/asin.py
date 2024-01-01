import numpy as np


class Arcsin:
    def __init__(self, x):
        self.x = x
        self.data = np.arcsin(x)

    def __call__(self, grad):
        return (grad / np.sqrt(1 - self.x ** 2),)
