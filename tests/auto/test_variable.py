import pytest
from auto.variable import Variable
from auto.math.add import AddBackward


class TestVariable:
    def test_sanity(self):
        a = Variable(5, True)
        b = Variable(10, True)
        assert isinstance(a + b, Variable)
        assert (a + b).data == 15
        assert (a + b).grad == 0
        assert (a + b).children == [a, b]
        assert isinstance((a + b).grad_fn, AddBackward)

    def test_addition(self):
        a = Variable(5, True)
        b = Variable(10, True)
        c = a + b
        assert c.data == 15
        c.backward()
        assert a.grad == 1
        assert b.grad == 1

    def test_multiplication(self):
        a = Variable(5, True)
        b = Variable(10, True)
        c = a * b
        assert c.data == 50
        c.backward()
        assert a.grad == 10
        assert b.grad == 5

    def test_sigmoid(self):
        a = Variable(5, True)
        b = a.sigmoid()
        assert b.data == pytest.approx(0.9933071490757268)
        b.backward()
        assert a.grad == pytest.approx(0.0066480566707786)

    def test_relu(self):
        a = Variable(1, requires_grad=True)
        b = a.relu()
        assert b.data == 1
        b.backward()
        assert b.grad == 1

        a.set(-1)
        b = a.relu()
        assert b.data == 0
        b.backward()
        assert b.grad == 1
