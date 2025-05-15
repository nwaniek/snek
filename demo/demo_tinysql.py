#!/usr/bin/env python

# be able to run this without actually installing snek
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snek import DependencyManager
from dataclasses import dataclass, is_dataclass, field, asdict, make_dataclass, fields

import tinysql


def _serialize(self, name: str, unique_id: str):
    if not hasattr(self, '_tinysql_proxy_class'):
        raise RuntimeError("Class not decorated with extended_db_table")
    proxy_cls = self._tinysql_proxy_class
    tspec     = self._tinysql_proxy_spec
    context   = proxy_cls._tinysql_context

    ## Inject the ID dynamically
    proxy_instance = proxy_cls(id=unique_id, **asdict(self))
    tinysql.insert(context, asdict(proxy_instance), tspec=tspec)


@classmethod
def _deserialize(cls, name: str, unique_id: str):
    if not hasattr(cls, '_tinysql_proxy_class'):
        raise RuntimeError("Class not decorated with extended_db_table")
    proxy_cls = cls._tinysql_proxy_class
    context   = proxy_cls._tinysql_context

    cond = tinysql.Equals("id", unique_id)
    results = list(tinysql.select(context, proxy_cls, cond))
    if not results:
        return None

    proxy_obj = results[0]
    # Strip the 'id' field and reconstruct original
    data = {k: getattr(proxy_obj, k) for k in proxy_obj.__dataclass_fields__ if k != "id"}
    return cls(**data)


def _augment_class(cls, tablename, primary_keys):
    # extract fields from the original class
    base_fields = [(f.name, f.type) for f in fields(cls)]

    # dynamically create proxy class with added field for the unique id
    extended_fields = [('id', str)] + base_fields
    proxy_cls = make_dataclass(cls.__name__ + "_TinySQLProxy", extended_fields)

    # decorate proxy class with db_table
    proxy_cls = tinysql.db_table(tablename if tablename != '' else cls.__name__, primary_keys=primary_keys)(proxy_cls)

    # register proxy class on the original class, and add de/serialization methods
    setattr(cls, "_tinysql_proxy_class", proxy_cls)
    setattr(cls, "_tinysql_proxy_spec", getattr(proxy_cls, "_tinysql_tspec"))
    setattr(cls, "serialize",   _serialize)
    setattr(cls, "deserialize", _deserialize)

    return cls


#def extended_db_table(_cls=None, tablename='', primary_keys=["id"]):
def db_cached(_cls=None, *, tablename='', primary_keys=['id']):
    def wrap(cls):
        return _augment_class(cls, tablename, primary_keys)

    if _cls is None:
        return wrap
    return wrap(_cls)


@dataclass
class AConfig:
    # subconfig for compute_A
    option_a_1: int = 101
    option_a_2: int = 12


@dataclass
class BConfig:
    # subconfig for compute_B
    option_b_1: float = 1.2
    option_b_2: int   = 71


@dataclass
class Configuration:
    config_a: AConfig = field(default_factory=lambda: AConfig())
    config_b: BConfig = field(default_factory=lambda: BConfig())



def attach(ctx):
    for value in ctx.registry.values():
        setattr(value.cls, '_tinysql_context', ctx)


@db_cached
@dataclass
class AResult:
    value: str


@db_cached
@dataclass
class BResult:
    value: str


@db_cached
@dataclass
class CResult:
    value: str


def register_pipeline(dm: DependencyManager, config: Configuration):

    @dm.target(provides='A', requires=['@demoinput.txt'], params=asdict(config.config_a))
    def compute_A(fpath, **kwargs) -> AResult:
        print("called compute_A()")
        print(f"Reading file {fpath}")
        return AResult("Result A (" + str(kwargs['option_a_1']) + ")")

    @dm.target(provides='C')
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

    context = tinysql.DatabaseContext('cache.sqlite')
    attach(context)
    with context:
        register_pipeline(dm, config)
        result = dm.make('B', use_cache=True, verbose=True)
        print(result)

