class Sum:
    def __init__(self, x):
        self.x = x
        self.data = x.sum()

    def __call__(self, grad):
        return (grad,)
