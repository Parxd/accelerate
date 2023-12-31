import numpy as np


class Sin:
    def __init__(self, x):
        self.x = x
        self.data = np.sin(x)

    def __call__(self, grad):
        return (grad * np.cos(self.x),)
