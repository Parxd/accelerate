import numpy as np


class Mean:
    def __init__(self, x):
        self.x = x
        self.data = np.mean(x)

    def __call__(self, grad):
        return (grad / self.x.size,)
