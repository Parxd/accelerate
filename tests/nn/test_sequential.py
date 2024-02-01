import numpy as np
import nn
from core.tensor import Tensor


class TestSequential:
    def test_sequential_1(self):
        seq = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 5),
            nn.Sigmoid()
        )
        assert len(seq) == 4

    def test_sequential_2(self):
        layers = [nn.Linear(2, 3),
                  nn.Sigmoid(),
                  nn.Linear(3, 2)]
        seq = nn.Sequential(*layers)
        assert len(seq) == 3

    def test_sequential_3(self):
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
