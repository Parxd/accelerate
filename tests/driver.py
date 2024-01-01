from core.tensor import Tensor


def main():
    l1_weights = Tensor.random((3, 5))
    print(l1_weights)
    return 0


if __name__ == "__main__":
    main()
