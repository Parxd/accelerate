import cupy as cp
import nn
from core.device import Device


def test1():
    X = nn.Linear(3, 5)
    for i in X.parameters():
        print()
        print(i)


def test2():
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Sigmoid(),
        nn.Linear(5, 2)
    )
    print()
    for i in model.parameters():
        print(i)


def test3():
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Sigmoid(),
        nn.Linear(5, 2)
    )
    model.to('cuda')
    for parameter in model.parameters():
        assert parameter.device is Device.GPU
        assert isinstance(parameter.data, cp.ndarray)


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
