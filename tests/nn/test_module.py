import nn


def test1():
    X = nn.Linear(3, 5)
    for i in X.parameters():
        print(i)


def test2():
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Sigmoid(),
        nn.Linear(5, 2)
    )
    for i in model.parameters():
        print(i)


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
