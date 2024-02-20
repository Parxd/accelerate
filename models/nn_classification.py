import numpy as np
import nn
import core

LR = 0.01
EPOCHS = 50
DATA_POINTS = 10000


def main():
    x_1 = np.random.rand(DATA_POINTS)
    x_2 = np.random.rand(DATA_POINTS)
    x_3 = np.random.rand(DATA_POINTS)
    X = core.Tensor(np.stack((x_1, x_2, x_3), axis=1))
    y = ((1.6 * x_1 + 0.1) - (0.7 * x_2 + 0.2) + (0.1 * x_3 + 0.3) + 0.2 * np.random.randn(DATA_POINTS) > 0.7).astype(float)
    new_y = np.zeros((DATA_POINTS, 2), dtype=int)
    new_y[:, 0] = y
    new_y[:, 1] = 1 - y

    print(y)
    print(new_y)

    criterion = nn.loss.BCELoss()
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Sigmoid(),
        nn.Linear(5, 5),
        nn.Sigmoid(),
        nn.Linear(5, 2),
        nn.Sigmoid()
    )

    # for i in range(EPOCHS):
    #     y_hat = model.forward(X)
    #     error = criterion(y_hat, y)
    #     error.backward()
    #     for param in model.parameters():
    #         param -= LR * param.grad
    #     model.zero_grad()
    #     if i % 1 == 0:
    #         print(f"iteration {i}: error={repr(error)}")
    # return 0


if __name__ == "__main__":
    main()
