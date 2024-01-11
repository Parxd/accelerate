from core.tensor import Tensor


class BCELoss:
    def __init__(self, reduction: str = 'mean'):
        if reduction.lower() not in ['none', 'mean', 'sum']:
            raise ValueError(f'{reduction} is not a valid value for reduction')
        self.reduction = reduction.lower()

    def __call__(self, output: Tensor, target: Tensor) -> Tensor:
        if self.reduction == 'none':
            return self._compute(output, target)
        elif self.reduction == 'mean':
            return self._compute(output, target).mean()
        elif self.reduction == 'sum':
            return self._compute(output, target).sum()

    @staticmethod
    def _compute(output: Tensor, target: Tensor) -> Tensor:
        return -(target * output.log() + (1 - target) * (1 - output).log())
