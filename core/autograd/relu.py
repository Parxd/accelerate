class ReLU:
    def __init__(self, x):
        self.x = x
        self.data = x * (x > 0)

    def __call__(self, grad):
        return (grad * ((self.x > 0) * 1),)
