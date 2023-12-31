class Neg:
    def __init__(self, x):
        self.x = x
        self.data = -x
    
    def __call__(self, grad):
        return (-grad,)
