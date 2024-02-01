from core.tensor import Tensor


class ReLU:
    def __init__(self):
        ...

    def __call__(self, data: Tensor) -> Tensor:
        return data.relu()

    def __str__(self):
        return "ReLU()"
