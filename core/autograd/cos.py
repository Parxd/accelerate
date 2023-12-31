import numpy as np


class Cos:
    def __init__(self, x):
        self.x = x
        self.data = np.cos(x)

    def __call__(self, grad):
        return (grad * -np.sin(self.x),)
