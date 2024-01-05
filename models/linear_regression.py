import numpy as np
import matplotlib.pyplot as plt
from core.tensor import Tensor

LR = 0.01
EPOCHS = 500
DATA_POINTS = 100
PLOT = True


def mse(y: np.ndarray, y_hat: Tensor) -> Tensor:
    return (y - y_hat).square().mean()


def main():
    x = np.arange(DATA_POINTS)
    x_normalized = (x - x.mean()) / x.std()
    y = 4 * x_normalized + np.random.uniform(0, 2, (DATA_POINTS,))

    weights = Tensor.random((1,), requires_grad=True)
    bias = Tensor.random((1,), requires_grad=True)

    for i in range(EPOCHS):
        y_hat = weights * x_normalized + bias
        error = mse(y, y_hat)
        error.backward()

        weights -= LR * weights.grad
        bias -= LR * bias.grad
        weights.clear_grad()
        bias.clear_grad()

        print(f"iteration {i}: error={error}")

    if PLOT:
        plt.plot(x_normalized, y)
        plt.plot(x_normalized, weights * x_normalized + bias)
        plt.show()
    return 0


if __name__ == "__main__":
    main()
