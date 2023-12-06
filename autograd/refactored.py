from __future__ import annotations
from typing import Callable, List
from dataclasses import dataclass


# Child -> struct w/ operand and its grad_fn
# N-ary operation -> list of N dependencies
@dataclass(frozen=True)
class Child:
    operand: Value
    grad_fn: Callable


class Value:
    def __init__(self):
        pass

