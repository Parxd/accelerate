import numpy as np
import matplotlib.pyplot as plt
from core import *
from nn import loss, Linear, Sigmoid

LR = 0.01
EPOCHS = 100
DATA_POINTS = 1000


def main():
    layer1 = Linear(3, 8)
    act1 = Sigmoid()
    layer2 = Linear(8, 1)
    act2 = Sigmoid()

    x_1 = np.random.rand(DATA_POINTS)
    x_2 = np.random.rand(DATA_POINTS)
    x_3 = np.random.rand(DATA_POINTS)

    criterion = loss.BCELoss

    return 0


if __name__ == "__main__":
    main()
