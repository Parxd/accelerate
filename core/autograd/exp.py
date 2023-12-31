import numpy as np


class Exp:
    def __init__(self, x):
        self.x = x
        self.data = np.exp(x)
    
    def __call__(self, grad):
        return (grad * self.data,)
