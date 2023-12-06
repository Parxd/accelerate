from autograd.autograd import Value
from unittest import TestCase


class SanityTests(TestCase):
    def test_add(self):
        pass

    def test_sub(self):
        pass

    def test_mul(self):
        pass

    def test_sigmoid(self):
        pass

    def test_combined(self):
        a = Value(5.1340, requires_grad=True)
        b = Value(2.3041, requires_grad=True)
        c = Value(2.6012, requires_grad=True)
        d = Value(6.7023, requires_grad=True)
        e = Value(1.2845, requires_grad=True)
        f = (a + b + c * e + d).sigmoid()
        f.backward()

    def test_constants(self):
        pass


def main():
    # a = 5
    # b = Value(3)
    # c = a * b
    # c.backward()
    # print(b._grad)
    return


if __name__ == "__main__":
    main()
