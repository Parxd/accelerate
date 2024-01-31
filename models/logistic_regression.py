import numpy as np
import matplotlib.pyplot as plt
from core.tensor import Tensor
from nn.loss import BCELoss

LR = 0.1
EPOCHS = 500
DATA_POINTS = 100
PLOT = True


def main():
    X = np.random.randn(DATA_POINTS,)
    y = ((1.6 * X.squeeze() + 1) + 0.2 * np.random.randn(DATA_POINTS,) > 1.5).astype(float)

    criterion = BCELoss()
    weight = Tensor(np.random.randn(1), requires_grad=True)
    bias = Tensor(np.random.randn(1), requires_grad=True)

    for i in range(EPOCHS):
        y_hat = (weight * Tensor(X) + bias).sigmoid()

        error = criterion(y_hat, Tensor(y))
        error.backward()

        weight -= LR * weight.grad
        bias -= LR * bias.grad
        weight.clear_grad()
        bias.clear_grad()

        if i % 10 == 0:
            print(f"iteration {i}: error={error}")

    if PLOT:
        final = (weight * Tensor(X) + bias).sigmoid()
        plt.scatter(X, y, color='blue')
        plt.scatter(X, final, color='orange')
        plt.show()
    return 0


if __name__ == "__main__":
    main()
