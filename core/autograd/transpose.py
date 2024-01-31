import numpy as np


class Transpose:
    def __init__(self, x):
        self.x = x
        self.data = np.transpose(x)

    def __call__(self, grad):
        return (grad.transpose(),)
