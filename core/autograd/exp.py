import numpy as np


class Exp:
    def __init__(self, mid):
        self.mid = mid
        self.data = np.exp(mid)
    
    def __call__(self, grad):
        return (grad * self.data,)
