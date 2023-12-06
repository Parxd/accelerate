from autograd import Value
from unittest import TestCase


class SanityTests(TestCase):
    def test_addition(self):
        pass
    
    def test_mul(self):
        pass
    
    def test_sigmoid(self):
        pass
    
    def test_combined(self):
        pass


def main():
    # a = ag.Value(5.1340, requires_grad=True)
    # b = ag.Value(2.3041 ,requires_grad=True)
    
    # c = ag.Value(2.5012, requires_grad=True)
    # d = ag.Value(6.7023, requires_grad=True)

    # f = ((a + b) + (c + d)).sigmoid()
    # f.backward()
    # print(f._data)
    
    a = Value(5, requires_grad=True)
    b = Value(3, requires_grad=True)
    d = Value(12, requires_grad=True)
    e = Value(2, requires_grad=True)
    c = a * (b + d) * e
    c.backward()
    print(e._grad)


if __name__ == "__main__":
    main()
