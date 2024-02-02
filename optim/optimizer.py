from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, params, lr=0.001, weight_decay=0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

