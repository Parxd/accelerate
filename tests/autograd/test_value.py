from autograd.value import Value


class TestValue:
    def test_attr(self):
        a = Value(5)
        assert a.data == 5
        assert a.grad == 0
        assert a.requires_grad is False

    def test_add(self):
        a = Value(5)
        b = Value(6)
        assert isinstance(a + b, Value)
