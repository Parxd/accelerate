import numpy as np
import matplotlib.pyplot as plt
from core import *
from nn import Linear, Sigmoid
from nn.loss import MSELoss

LR = 0.01
EPOCHS = 100
DATA_POINTS = 1000


def main():
    layer1 = Linear(3, 8)
    act1 = Sigmoid()
    layer2 = Linear(8, 1)

    x_1 = np.random.rand(DATA_POINTS)
    x_2 = np.random.rand(DATA_POINTS)
    x_3 = np.random.rand(DATA_POINTS)
    X = Tensor(np.stack((x_1, x_2, x_3), axis=1))
    noise = np.random.randn(DATA_POINTS) * 0.1
    y = -0.6 * x_1 - 1.2 * x_2 + 0.7 * x_3 + noise

    criterion = MSELoss()
    for i in range(EPOCHS):
        y_hat = layer2(act1(layer1(X)))
        loss = criterion(y_hat, Tensor(y))
        loss.backward()

        # this is pretty bad, but just for sake of demonstration
        # ---
        # we also need something like torch's "with torch.no_grad()" to prevent gradient modification from
        # subtracting the actual gradient
        layer2._w -= LR * layer2._w.grad
        layer2._b -= LR * layer2._b.grad
        layer2._w.clear_grad()
        layer2._b.clear_grad()

        layer1._w -= LR * layer1._w.grad
        layer1._b -= LR * layer1._b.grad
        layer1._w.clear_grad()
        layer1._b.clear_grad()

        if i % 10 == 0:
            print(f"iteration {i}: error={loss}")

    return 0


if __name__ == "__main__":
    main()
