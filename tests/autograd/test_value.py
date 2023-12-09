from autograd import Value
import pytest


class TestValue:
    def test_attr(self):
        a = Value(5)
        assert a._data == 5
        assert a._grad == 0
        assert a.requires_grad is False

        b = Value(6, requires_grad=True)
        assert b.requires_grad is True
        assert (a + b).requires_grad is True

    def test_types(self):
        a = Value(5, requires_grad=True)
        b = Value(6)
        assert isinstance(a + b, Value)

    def test_expr(self):
        a = Value(5.1293, requires_grad=True)
        b = Value(153.43203, requires_grad=True)
        c = Value(2.105, requires_grad=True)
        d = (a + b) * c
        d.backward()
        assert c._grad == pytest.approx((a + b)._data)
