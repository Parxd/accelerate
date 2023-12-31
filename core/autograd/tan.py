import numpy as np


class Tan:
    def __init__(self, x):
        self.x = x
        self.data = np.tan(x)

    def __call__(self, grad):
        return (grad / np.cos(self.x) ** 2,)
