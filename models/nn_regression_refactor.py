import numpy as np
from core import *
import nn

LR = 0.01
EPOCHS = 100
DATA_POINTS = 1000


def main():
    x_1, x_2, x_3 = np.random.rand(DATA_POINTS), np.random.rand(DATA_POINTS), np.random.rand(DATA_POINTS)
    X = Tensor(np.stack((x_1, x_2, x_3), axis=1))
    noise = np.random.randn(DATA_POINTS) * 0.1
    y = 0.2 * x_1 - 1.2 * x_2 - 0.2 * x_3 + noise

    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1)
    )
    criterion = nn.loss.MSELoss()

    for i in range(EPOCHS):
        y_hat = model.forward(X)
        error = criterion(y_hat, Tensor(y))
        error.backward()
        for param in model.parameters():
            param -= LR * param.grad
        model.zero_grad()
        if i % 10 == 0:
            print(f"iteration {i}: error={repr(error)}")
    return 0


if __name__ == "__main__":
    main()
