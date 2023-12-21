from .shared_types import Tensor


def add(a: Tensor,
        b: Tensor):
    return a + b


# consider why we use a class instead of a function just for backpropagation
# the class can ENCAPSULATE data, while the function cannot
# what can the class encapsulate? the operand (child) data and the resulting (parent) data
