import time
import numpy as np
from core.tensor import Tensor
import nn

LR = 0.08
EPOCHS = 100
DATA_POINTS = 10000


def main():
    x_1, x_2, x_3 = np.random.rand(DATA_POINTS), np.random.rand(DATA_POINTS), np.random.rand(DATA_POINTS)
    X = Tensor(np.stack((x_1, x_2, x_3), axis=1))
    noise = np.random.randn(DATA_POINTS) * 0.1
    y = 0.2 * x_1 - 1.2 * x_2 - 0.2 * x_3 + noise

    model = nn.Sequential(
        nn.Linear(3, 10),
        nn.Sigmoid(),
        nn.Linear(10, 1)
    )
    criterion = nn.loss.MSELoss()

    # test training with GPU
    model.to('cuda')
    X.to('cuda')
    y = Tensor(y, device='cuda')
    start = time.time()
    for i in range(EPOCHS):
        y_hat = model.forward(X)
        error = criterion(y_hat, y)
        error.backward()
        for param in model.parameters():
            param -= LR * param.grad
        model.zero_grad()
        if i % 1 == 0:
            print(f"iteration {i}: error={repr(error)}")
    dur = time.time() - start
    print(dur)
    return 0


if __name__ == "__main__":
    main()
