from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GradientContext:
    op1_data: int | float = 0
    op2_data: int | float | None = 0
    parent_data: int | float = 0
    parent_grad: int | float = 0
