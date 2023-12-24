import numpy as np
from tensor import Tensor

X = Tensor([[1, 2], [3, 4]], requires_grad=True)
Y = X.sum()
Y.data + np.array(1)

print(X.grad.data)
print(X.grad.data.dtype)
Tensor(1).