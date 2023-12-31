class Square:
    def __init__(self, x):
        self.x = x
        self.data = x ** 2

    def __call__(self, grad):
        return (grad * 2 * self.x,)
