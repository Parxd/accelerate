from autograd import Value
import pytest


class TestValue:
    def test_attr(self):
        pass

    def test_types(self):
        pass

    def test_const_add(self):
        a = Value(2.591, requires_grad=True)

        d = 15 + a + 10
        d.backward()
        assert a._grad == 1

    def test_const_mul(self):
        a = Value(2.591, requires_grad=True)

        b = a * 5
        assert b._data == pytest.approx(12.955)
        b.backward()
        assert a._grad == 5
        a.clear_grad()

        c = 10 * a
        assert c._data == pytest.approx(25.910)
        c.backward()
        assert a._grad == 10

    def test_reverse(self):
        a = Value(2.591, requires_grad=True)

        b = 5 - 5 * a
        assert b._data == pytest.approx(-7.955)
        b.backward()
        assert a._grad == -5

    def test_negation(self):
        pass

    def test_sigmoid(self):
        a = Value(1, requires_grad=True)
        assert a.sigmoid()._data == pytest.approx(0.731058578630074)

        a.sigmoid().backward()
        assert a._grad == 0

    def test_relu(self):
        a = Value(-2, requires_grad=True)
        assert a.relu()._data == 0

        a.relu().backward()
        assert a._grad == 0
