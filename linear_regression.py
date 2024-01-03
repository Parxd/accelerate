import numpy as np
import matplotlib.pyplot as plt
from core.tensor import Tensor


LR = 0.01
DATA_POINTS = 15


def mse(y: np.ndarray, y_hat: Tensor) -> Tensor:
    return (y - y_hat).square().mean()


def main():
    x = np.arange(DATA_POINTS)
    y = 1.5 * x + np.random.randn(DATA_POINTS)

    weights = Tensor.random((1,), requires_grad=True)
    bias = Tensor.random((1,), requires_grad=True)

    for i in range(50):
        y_hat = weights * x + bias
        error = mse(y, y_hat)
        error.backward()
        weights -= LR * weights.grad
        bias -= LR * bias.grad
        weights.clear_grad()
        bias.clear_grad()
        print(f"iteration {i}: error={error}")

    plt.plot(x, y)
    plt.plot(x, weights * x + bias)
    plt.show()
    return 0


if __name__ == "__main__":
    main()
