class Neg:
    def __init__(self, mid):
        self.mid = mid
        self.data = -mid
    
    def __call__(self, grad):
        return (-grad,)
