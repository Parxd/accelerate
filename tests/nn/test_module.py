import numpy as np
import nn
from core.tensor import Tensor


class TestModule:
    def test_module_1(self):
        seq = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid()
        )
        assert len(seq) == 2
        assert isinstance(seq[0], nn.Linear)
        assert isinstance(seq[1], nn.Sigmoid)

    def test_module_2(self):
        layers = [nn.Linear(2, 3),
                  nn.Sigmoid()]
        seq = nn.Sequential(*layers)
        assert len(seq) == 2
        assert isinstance(seq[0], nn.Linear)
        assert isinstance(seq[1], nn.Sigmoid)

    def test_module_3(self):
        seq = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1)
        )
        data = np.array([[1, 2],
                         [3, 4]])
        fwd = seq.forward(Tensor(data))
        assert isinstance(fwd, Tensor)
        assert fwd._shape == (2, 1)

    def test_module_4(self):
        seq = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1)
        )
        data = np.array([[1, 2],
                         [3, 4]])
        fwd = seq.forward(Tensor(data))
        target = np.array([[0.15],
                           [0.75]])
        criterion = nn.loss.MSELoss()
        error = criterion(fwd, Tensor(target))
        error.backward()

        # ensure gradients are filled out
        assert seq[0]._w.grad != Tensor(np.zeros_like(seq[0]._w.data))
        assert seq[0]._b.grad != Tensor(np.zeros_like(seq[0]._b.data))
        seq.zero_grad()
        # ensure gradients are zeroed out
        assert seq[0]._w.grad == Tensor(np.zeros_like(seq[0]._w.data))
        assert seq[0]._b.grad == Tensor(np.zeros_like(seq[0]._b.data))

    def test_module_5(self):
        seq = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1)
        )
        # 2 Linear classes w/ 2 parameters each, so 4 parameters
        assert len(seq.parameters) == 4
        assert isinstance(seq.parameters[0], Tensor)
