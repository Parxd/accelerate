from autograd.deprecated.old import Value


def main():
    a = Value(5.123, requires_grad=True)
    b = Value(7.213, requires_grad=True)
    c = a * b
    c.backward()
    print(a._grad)

    return


if __name__ == "__main__":
    main()
