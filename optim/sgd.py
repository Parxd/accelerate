from optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, weight_decay=0):
        super(SGD, self).__init__(params, lr, weight_decay)

    def step(self):
        ...
