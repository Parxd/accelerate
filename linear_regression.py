import numpy as np
import matplotlib.pyplot as plt
from core.tensor import Tensor


def mse(y: Tensor, y_hat: Tensor) -> Tensor:
    return (y - y_hat).square().mean()


def main():
    weights = Tensor.random((1,))
    bias = Tensor.random((1,))

    np.random.seed = 0
    x = np.arange(100)
    noise = np.random.uniform(-10, 10, size=(100,))

    y = 2 * x + noise
    y_hat = weights * x + bias
    error = mse(y, y_hat)
    error.backward()
    

    plt.plot(x, y)
    plt.plot(x, y_hat)
    plt.show()
    return 0


if __name__ == "__main__":
    main()
