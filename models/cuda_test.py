import time
from core import Tensor
import cupy as cp
import numpy as np


def main():
    A = Tensor(np.random.randn(10000, 200), requires_grad=True)
    B = Tensor(np.random.randn(200, 10000), requires_grad=True)
    A.to('cuda')
    B.to('cuda')
    for _ in range(200):
       C = (A @ B).mean()
       C.backward()


if __name__ == "__main__":
    main()
