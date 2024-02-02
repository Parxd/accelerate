import numpy as np
import nn
from core.tensor import Tensor


def test1():
    X = nn.Linear(3, 5)
    print()
    for i in X.parameters():
        print(i)


def test2():
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Sigmoid(),
        nn.Linear(5, 2)
    )
    print()
    print(model)


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
