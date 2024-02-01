import numpy as np

import nn
from core.tensor import Tensor
from nn import *
from nn.loss import *


class TestSequential:
    def test_sequential_1(self):
        seq = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid(),
            nn.Linear(3, 5),
            nn.Sigmoid()
        )
        print()
        print(seq)
