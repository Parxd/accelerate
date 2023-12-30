class Sum:
    def __init__(self, mid):
        self.mid = mid
        self.data = mid.sum()

    def __call__(self, grad):
        return (grad,)
