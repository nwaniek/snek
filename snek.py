#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2025 Nicolai Waniek <n@rochus.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from typing import List, Dict, Optional, Callable, Any, get_type_hints
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
import inspect
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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


def toposort(node: Node) -> List[Node]:
   visited = set()
   ordered = []

   def visit(n: Node):
       if n.unique_id in visited:
           return
       visited.add(n.unique_id)
       for dep in n.dependencies:
           visit(dep)
       ordered.append(n)

   visit(node)
   return ordered


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


    def build_graph(self, name, verbose=False) -> Node:
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


    def resolve(self, node: Node,  use_cache: bool = False, verbose: bool = False) -> Any:
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


    def resolve_parallel(self, node: Node):
        ordered_nodes = toposort(node)
        futures = {}

        # for n in ordered_nodes:
        #     deps = []
        #     if de.is_file:


        #     deps = [futures[dep.unique_id] for dep in n.dependencies if not d.is_file ]


    #def resolve_parallel(self, node: Node, use_cache: bool = False):
    #    ordered_nodes = toposort(node)
    #    futures = {}
    #    resolved = {}
    #    lock = Lock()

    #    print("order: ")
    #    for n in ordered_nodes:
    #        print(n.name)

    #    print("---")

    #    def resolve_node(n):
    #        # check if this is already in cache
    #        with lock:
    #            if n.unique_id in self.cache:
    #                return resolved[n.unique_id]

    #        if n.is_file:
    #            resolved[n.unique_id] = n.fpath
    #            return n.fpath

    #        args = []
    #        for dep in n.dependencies:
    #            with lock:
    #                args.append(resolved[dep.unique_id] if not dep.is_file else dep.fpath)

    #        print(n.name, "args", args)

    #        params      = n.params.copy()
    #        sig         = inspect.signature(n.func)
    #        param_names = list(sig.parameters)
    #        bound_args  = {}
    #        for i, arg in enumerate(args):
    #            if i < len(param_names):
    #                bound_args[param_names[i]] = arg
    #        bound_args.update(params)

    #        print(n.name, "bound_args", bound_args)

    #        result = n.func(**bound_args)
    #        print(result)

    #        # if use_cache and n.cacheable and n.serializer:
    #        #     n.serializer(result, n.name, n.unique_id)

    #        with lock:
    #            resolved[n.unique_id] = result

    #        return result

    #    with ThreadPoolExecutor() as executor:
    #        for n in ordered_nodes:
    #            futures[n.unique_id] = executor.submit(resolve_node, n)

    #        for unique_id, future in futures.items():
    #            self.cache[unique_id] = future.result()
    #            # print(unique_id, future)


    #    # def resolve_node(n):
    #    #     # Check cache
    #    #     with lock:
    #    #         if n.unique_id in resolved:
    #    #             return resolved[n.unique_id]

    #    #     args = []
    #    #     for dep in n.dependencies:
    #    #         with lock:
    #    #             args.append(resolved[dep.unique_id] if not dep.is_file else dep.fpath)

    #    #     params = n.params.copy()
    #    #     sig = inspect.signature(n.func)
    #    #     param_names = list(sig.parameters)
    #    #     bound_args = {param_names[i]: args[i] for i in range(len(args))}
    #    #     bound_args.update(params)
    #    #     result = n.func(**bound_args)

    #    #     if use_cache and n.cacheable and n.serializer:
    #    #         n.serializer(result, n.name, n.unique_id)

    #    #     with lock:
    #    #         resolved[n.unique_id] = result
    #    #     return result

    #    # with ThreadPoolExecutor() as executor:
    #    #     for n in ordered_nodes:
    #    #         futures[n.unique_id] = executor.submit(resolve_node, n)

    #    #     for unique_id, future in futures.items():
    #    #         self.cache[unique_id] = future.result

    #    # return self.cache[node.unique_id]


    def make(self, name, verbose: bool = False, use_cache: bool = True):
        node   = self.build_graph(name, verbose)
        result = self.resolve(node, use_cache, verbose)
        return result


    def make_parallel(self, name, verbose: bool = False, use_cache: bool = True):
        node = self.build_graph(name, verbose)

        print('\nresolve_parallel')
        print('----------------')
        result = self.resolve_parallel(node, use_cache)
        return result
