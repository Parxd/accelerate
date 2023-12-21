import numpy as np
from core.tensor import Tensor


def zeros(shape):
    return Tensor(np.zeros(shape),
                  )


def ones(shape):
    return Tensor(np.ones(shape))
