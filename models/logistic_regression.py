import numpy as np
import matplotlib.pyplot as plt
from core.tensor import Tensor
from nn.loss import *

LR = 0.1
EPOCHS = 100
DATA_POINTS = 100
PLOT = False


def main():
    X = np.random.randn(DATA_POINTS,)
    y = ((1.6 * X.squeeze() + 1) + 0.2 * np.random.randn(DATA_POINTS,) > 1.5).astype(float)

    criterion = BCELoss()
    weight = Tensor(np.random.randn(1), requires_grad=True)
    bias = Tensor(np.random.randn(1), requires_grad=True)

    for _ in range(EPOCHS):
        y_hat = (weight * Tensor(X) + bias).sigmoid()

        loss = criterion(y_hat, Tensor(y))
        loss.backward()

        weight -= LR * weight.grad
        bias -= LR * bias.grad
        weight.clear_grad()
        bias.clear_grad()
        print(loss)

    final = (weight * Tensor(X) + bias).sigmoid()
    plt.scatter(X, y, color='blue')
    plt.scatter(X, final, color='orange')
    plt.xlabel('Independent Feature')
    plt.ylabel('Dependent Feature (Binary)')
    plt.title('Generated Data for Logistic Regression')
    plt.show()

    return 0


if __name__ == "__main__":
    main()
