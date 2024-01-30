from core.tensor import Tensor


class BCELoss:
    def __init__(self, reduction: str = 'mean', eps: float = 1e-12):
        if reduction.lower() not in ['none', 'mean', 'sum']:
            raise ValueError(f'{reduction} is not a valid value for reduction')
        self.reduction = reduction.lower()
        self.eps = eps

    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        if self.reduction == 'none':
            return self._compute(output, target)
        elif self.reduction == 'mean':
            return self._compute(output, target).mean()
        elif self.reduction == 'sum':
            return self._compute(output, target).sum()

    # epsilon to stabilize computations
    def _compute(self, output: Tensor, target: Tensor) -> Tensor:
        return -(target * (output + self.eps).log() + (1 - target) * (1 - (output + self.eps)).log())
