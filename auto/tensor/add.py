def add(a, b):
    return AddBackward(a,
                       b,
                       a + b,
                       0)


class AddBackward:
    def __init__(self, left_child, right_child, parent, parent_grad):
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent
        self.parent_grad = parent_grad

    def compute_grad(self):
        return
