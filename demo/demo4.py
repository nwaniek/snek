#!/usr/bin/env python

# be able to run this without actually installing snek
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snek import DependencyManager
from dataclasses import dataclass

dm = DependencyManager()

@dataclass
class AResult:
    value: str

@dataclass
class BResult:
    value: str


@dataclass
class CResult:
    value: str

@dm.target(provides='A', cacheable=False)
def compute_A() -> AResult:
    print("called compute_A()")
    return AResult("Result of A()")

@dm.target(provides='C', cacheable=False)
def compute_C() -> CResult:
    print("called compute_C()")
    return CResult("C Result")

@dm.target(provides='B', requires=['A', 'C'], cacheable=False)
def compute_B(A, C) -> BResult:
    print("called compute_B()")
    return BResult("Result B following " + A.value + " " + C.value)

if __name__ == "__main__":
    result = dm.make('B')
    print(result)

