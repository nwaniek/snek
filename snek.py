#!/usr/bin/env python

from typing import List, Dict, Optional, Callable, Any, get_type_hints
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
import inspect
import hashlib

__version__ = '0.1.1'

try:
    import orjson
    def __dumps(obj: Any) -> str:
        return orjson.dumps(obj, default=str, option=orjson.OPT_SORT_KEYS).decode()

except ImportError:
    import json
    def __dumps(obj: Any) -> str:
        return json.dumps(obj, default=str, sort_keys=True)

def hash_file(file_path: Path) -> str:
    """Generate a hash based on the content of a file."""
    with open(file_path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def hash_obj(obj: Any) -> str:
    """Generate a hash from a given object."""
    if isinstance(obj, Path):  # Special case: if it's a file, hash its contents
        return hash_file(obj)
    obj_json = __dumps(obj)
    return hashlib.sha1(obj_json.encode()).hexdigest()


@dataclass
class Node:
    name         : str
    unique_id    : str
    cacheable    : bool
    dependencies : List["Node"]
    params       : Dict[str, Any]
    func         : Optional[Callable]
    return_type  : Any
    serializer   : Optional[Callable[[Any, str, str], None]]
    deserializer : Optional[Callable[[str, str], Any]]
    fpath        : Optional[Path | None]
    is_file      : bool


def resolve_file_dep(name: str) -> Path:
    assert name.startswith('@')
    return Path(name[1:])


class DependencyManager:

    def __init__(self, cache_dir = 'cache'):
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.cache = {}
        self.cache_dir = Path(cache_dir)


    def target(self,
               provides: Optional[str]=None,
               requires: Optional[List[str]]=None,
               params: Optional[Dict[str, Any]] = None,
               strict_param_check: bool=False,
               cacheable:bool=True,
               return_type=None,
               serializer=None,
               deserializer=None):
        params = params or {}
        def decorator(func):
            name          = func.__name__ if provides is None else provides
            if name in self.registry:
                raise ValueError(f"Duplicate target '{name}'")

            hints         = get_type_hints(func)
            rtype         = return_type or hints.get('return', None)
            signature     = inspect.signature(func)
            if strict_param_check:
                invalid_keys = set(params) - set(signature.parameters)
                if invalid_keys:
                    raise TypeError(f"{name}: Unrecognized config keys: {invalid_keys}")

            requires_list = [param.name for param in signature.parameters.values() if param.kind not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]]
            self.registry[name] = {
                "func": func,
                "cacheable": cacheable,
                "return_type": rtype,
                "serializer": serializer,
                "deserializer": deserializer,
                "param_names": set(signature.parameters),
                "requires": requires if requires is not None else requires_list,
                "params": params,
            }

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorator


    def build_graph(self, name, verbose=False):
        if name not in self.registry:
            raise ValueError(f"Target {name} not known")

        info         = self.registry[name]
        func         = info['func']
        return_type  = info['return_type']
        dependencies = info['requires']
        params       = info['params']
        serializer   = info["serializer"]
        deserializer = info["deserializer"]
        cacheable    = info["cacheable"]
        if not serializer and hasattr(return_type, "serialize"):
            serializer = lambda obj, name, unique_id: obj.serialize(name, unique_id)
        if not deserializer and hasattr(return_type, "deserialize"):
            deserializer = lambda name, unique_id: return_type.deserialize(name, unique_id)

        # determine unique ID of this node given name and parent hashes
        dep_nodes = []
        parent_hashes = []
        for dep in dependencies:
            if isinstance(dep, str) and dep.startswith('@'):
                fpath = resolve_file_dep(dep)
                if not fpath.exists():
                    raise FileNotFoundError(f"File dependency '{fpath}' not found!")
                fhash = hash_file(fpath)
                unique_id = "file_" + fhash
                node = Node(dep[1:], "file_" + fhash, False, [], {}, None, None, None, None, fpath, True)
                dep_nodes.append(node)
                parent_hashes.append(fhash)
            else:
                subnode = self.build_graph(dep, verbose)
                dep_nodes.append(subnode)
                parent_hashes.append(subnode.unique_id)

        parent_hashes = ''.join(parent_hashes)
        hash_input = {
            "name":          name,
            "params":        params,
            "parent_hashes": parent_hashes,
        }
        unique_id = hash_obj(hash_input)
        cache_path = self.cache_dir / (unique_id + '.npz')
        if verbose:
            print(f"Resolved: {name} {unique_id}")
        return Node(name, unique_id, cacheable, dep_nodes, params, func, return_type, serializer, deserializer, cache_path, False)


    def resolve(self, node: Node,  use_cache: bool = False, verbose: bool = False):
        if node.unique_id in self.cache:
            if verbose:
                print(f"Using cached {node.name}")
            return self.cache[node.unique_id]

        if use_cache and node.cacheable and node.deserializer:
            result = node.deserializer(node.name, node.unique_id)
            if result is not None:
                if verbose:
                    print(f"Loaded {node.name} from cache")
                self.cache[node.unique_id] = result
                return result

        resolved_deps = []
        for dep in node.dependencies:
            if dep.is_file:
                resolved_deps.append(dep.fpath)
            else:
                resolved_deps.append(self.resolve(dep, use_cache))

        if not node.func:
            raise RuntimeError(f"No Callable for node {node.name}")

        args        = resolved_deps
        params      = node.params.copy()
        sig         = inspect.signature(node.func)
        param_names = list(sig.parameters)
        bound_args  = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                bound_args[param_names[i]] = arg

        bound_args.update(params)
        result = node.func(**bound_args)
        self.cache[node.unique_id] = result
        if use_cache and node.cacheable and node.serializer:
            node.serializer(result, node.name, node.unique_id)

        return result


    def make(self, name, verbose: bool = True, use_cache: bool = False):
        node   = self.build_graph(name, verbose)
        result = self.resolve(node, use_cache, verbose)
        return result

