import numpy as np
import matplotlib.pyplot as plt
from core.tensor import Tensor
from nn.loss import *


LR = 0.01
EPOCHS = 500
DATA_POINTS = 100
PLOT = True


def main():
    x = np.arange(1, 50)
    y = (4 + 3 * x + np.random.randn(DATA_POINTS, 1) > 6).astype(int)
    print(y)

    loss = BCELoss()
    weights = Tensor.random((1,), requires_grad=True)
    bias = Tensor.random((1,), requires_grad=True)

    # for i in range(EPOCHS):
    #     y_hat = (weights * x + bias).sigmoid()
    #     error = loss(y_hat, y)
    #     error.backward()
    #
    #     weights -= LR * weights.grad
    #     bias -= LR * bias.grad
    #     weights.clear_grad()
    #     bias.clear_grad()
    #
    #     print(error)

    # if PLOT:
    #     plt.plot(x)
    #     plt.show()

    return 0


if __name__ == "__main__":
    main()
