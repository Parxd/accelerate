import numpy as np


class Log:
    def __init__(self, x):
        self.x = x
        self.data = np.log(x)

    def __call__(self, grad):
        return (grad / self.x,)
