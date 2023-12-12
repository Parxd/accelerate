from auto.variable import Variable, grad_fn_mapping
from auto.math import add


class TestVariable:
    def test_sanity(self):
        a = Variable(5, 0, True,None, None)
        b = Variable(10, 0, True, None, None)
        assert isinstance(a + b, Variable)
        assert (a + b).data == 15
        assert (a + b).grad == 0
        assert (a + b).children == [a, b]
        assert (a + b).grad_fn == grad_fn_mapping[add]

        c = a + b
        c.backward()
        assert a.grad == 1
        assert b.grad == 1
