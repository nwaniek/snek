#!/usr/bin/env python

# be able to run this without actually installing snek
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snek import DependencyManager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dcutils import dataclass_to_npz, npz_to_dataclass


@dataclass
class AConfig:
    # subconfig for compute_A
    option_a_1: int = 101
    option_a_2: int = 12


@dataclass
class BConfig:
    # subconfig for compute_B
    option_b_1: float = 1.2
    option_b_2: int   = 711


@dataclass
class Configuration:
    config_a: AConfig = field(default_factory=lambda: AConfig())
    config_b: BConfig = field(default_factory=lambda: BConfig())


class Serializable:
    def serialize(self, name: str, unique_id: str):
        fpath = Path('cache') / (name + "_" + unique_id + ".npz")
        dataclass_to_npz(self, fpath)

    @classmethod
    def deserialize(cls, name: str, unique_id: str):
        fpath = Path('cache') / (name + "_" + unique_id + ".npz")
        if fpath.exists():
            return npz_to_dataclass(cls, fpath)
        return None


@dataclass
class AResult(Serializable):
    value: str

@dataclass
class BResult(Serializable):
    value: str

@dataclass
class CResult(Serializable):
    value: str


def register_pipeline(dm: DependencyManager, config: Configuration):

    @dm.target(provides='A', requires=['@demoinput.txt'], params=asdict(config.config_a))
    def compute_A(fpath, **kwargs) -> AResult:
        print("called compute_A()")
        print(f"Reading file {fpath}")
        return AResult("Result A (" + str(kwargs['option_a_1']) + ")")

    @dm.target(provides='C', params=None)
    def compute_C() -> CResult:
        print("called compute_C()")
        return CResult("C Result")

    @dm.target(provides='B', requires=['A', 'C'], params={'config_b': config.config_b})
    def compute_B(A, C, config_b) -> BResult:
        print("called compute_B()")
        return BResult("Result B following " + A.value + " " + C.value)


if __name__ == "__main__":
    dm = DependencyManager()
    config = Configuration()

    register_pipeline(dm, config)
    result = dm.make_parallel('B', use_cache=True, verbose=True)
    print(result)

