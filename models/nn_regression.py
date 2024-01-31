import numpy as np
import matplotlib.pyplot as plt
from core.tensor import Tensor
from nn import Linear, Sigmoid
from nn.loss import MSELoss

LR = 0.1
EPOCHS = 10
DATA_POINTS = 500
PLOT = True


def main():
    layer1 = Linear(2, 8)
    act1 = Sigmoid()
    layer2 = Linear(8, 1)

    x_1 = np.random.rand(DATA_POINTS)
    x_2 = np.random.rand(DATA_POINTS)
    X = Tensor(np.stack((x_1, x_2), axis=1))
    noise = np.random.randn(DATA_POINTS) * 0.1
    y = 0.6 * x_1 + 1.2 * x_2 + noise

    criterion = MSELoss()
    for i in range(EPOCHS):
        y_hat = layer2(act1(layer1(X)))
        loss = criterion(y_hat, Tensor(y))
        loss.backward()

        # this is pretty bad, but just for sake of demonstration
        layer2._w -= LR * layer2._w.grad
        layer2._b -= LR * layer2._b.grad
        layer2._w.clear_grad()
        layer2._b.clear_grad()

        layer1._w -= LR * layer1._w.grad
        layer1._b -= LR * layer1._b.grad
        layer1._w.clear_grad()
        layer1._b.clear_grad()

        print(f"iteration {i}: error={loss}")

    final = layer2(act1(layer1(X)))
    if PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_1, x_2, y)
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        ax.set_zlabel('y')
        plt.show()

    return 0


if __name__ == "__main__":
    main()
