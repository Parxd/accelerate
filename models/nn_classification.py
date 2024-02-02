import numpy as np
import matplotlib.pyplot as plt
import nn
from core import *

LR = 0.01
EPOCHS = 50
DATA_POINTS = 10000


def main():
    x_1 = np.random.rand(DATA_POINTS)
    x_2 = np.random.rand(DATA_POINTS)
    x_3 = np.random.rand(DATA_POINTS)

    criterion = nn.loss.BCELoss()
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
        nn.Sigmoid()
    )

    return 0


if __name__ == "__main__":
    main()
