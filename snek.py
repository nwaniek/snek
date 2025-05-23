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
from concurrent.futures import ThreadPoolExecutor, Future

__version__ = '0.1.3'

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


def _run_task(func, deps, params):
    sig = inspect.signature(func)
    arguments = sig.parameters

    arg_names           = []
    var_positional_name = None
    var_keyword_name    = None

    for name, param in arguments.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            var_positional_name = name
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_name = name
        else:
            arg_names.append(name)

    # bind deps to named args if possible
    bound_positional = []
    bound_kwargs     = {}
    dep_index        = 0

    for name in arg_names:
        if dep_index < len(deps):
            bound_positional.append(deps[dep_index])
            dep_index += 1
        elif name in params:
            bound_kwargs[name] = params.pop(name)
        else:
            raise ValueError(f"Missing value for parameter '{name}'")

    # remaining deps go to *args (if it was provided)
    extra_deps = deps[dep_index:]
    extra_deps_to_kwargs = False

    if extra_deps:
        if var_positional_name:
            bound_positional.extend(extra_deps)
        elif var_keyword_name:
            extra_deps_to_kwargs = True
        else:
            raise TypeError("Too many dependencies provided")

    # remaining params go to **kwargs as well as remaining deps if no *args was
    # provided.
    if var_keyword_name:
        bound_kwargs.update(params)
        if extra_deps_to_kwargs:
            # make sure not to overwrite another param
            kw_var = 'args'
            i = 0
            while kw_var in params:
                kw_var = 'args' + str(i)
                i += 1
            bound_kwargs[kw_var] = extra_deps

    elif params:
        raise TypeError(f"Unexpected parameters: {list(params)}")

    return func(*bound_positional, **bound_kwargs)



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
                "func":         func,
                "cacheable":    cacheable,
                "return_type":  rtype,
                "serializer":   serializer,
                "deserializer": deserializer,
                "param_names":  set(signature.parameters),
                "requires":     requires if requires is not None else requires_list,
                "params":       params,
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
            print(f"Node: {name} {unique_id}")
        return Node(name, unique_id, cacheable, dep_nodes, params, func, return_type, serializer, deserializer, cache_path, False)


    def _retrieve_from_cache(self, node, use_cache:bool, verbose:bool):
        if node.unique_id in self.cache:
            if verbose:
                print(f"Using memory-cached {node.name}")
            return self.cache[node.unique_id]

        if use_cache and node.cacheable and node.deserializer:
            result = node.deserializer(node.name, node.unique_id)
            if result is not None:
                if verbose:
                    print(f"Loaded {node.name} from cache")
                self.cache[node.unique_id] = result
                return result

        return None


    def resolve(self, node: Node,  use_cache: bool = False, verbose: bool = False) -> Any:
        if result := self._retrieve_from_cache(node, use_cache, verbose):
            return result

        resolved_deps = []
        for dep in node.dependencies:
            if dep.is_file:
                resolved_deps.append(dep.fpath)
            else:
                resolved_deps.append(self.resolve(dep, use_cache))

        if not node.func:
            raise RuntimeError(f"No Callable for node {node.name}")

        result = _run_task(node.func, resolved_deps, node.params.copy())
        self.cache[node.unique_id] = result
        if use_cache and node.cacheable and node.serializer:
            node.serializer(result, node.name, node.unique_id)

        return result


    def find_dirty_nodes_toposorted(self, node: Node, use_cache: bool) -> list[Node]:
        visited      = set()
        topo_sorted  = []
        dirty_cache  = {}

        def dfs(n: Node) -> bool:
            if n.unique_id in visited:
                return dirty_cache[n.unique_id]
            visited.add(n.unique_id)

            if n.is_file:
                dirty_cache[n.unique_id] = False
                return False

            if n.unique_id in self.cache:
                dirty_cache[n.unique_id] = False
                return False

            if use_cache and n.cacheable and n.deserializer:
                obj = n.deserializer(n.name, n.unique_id)
                if obj is not None:
                    self.cache[n.unique_id] = obj
                    dirty_cache[n.unique_id] = False
                    return False

            # check dependencies recursively
            any_dep_dirty = False
            for dep in n.dependencies:
                if dfs(dep):
                    any_dep_dirty = True

            # This node is dirty either because children are or it itself needs
            # compute (it's not in the memory cache and deserialization didn't
            # succeed either)
            is_dirty = True if (n.func is not None) else False
            is_dirty = is_dirty or any_dep_dirty
            if is_dirty:
                topo_sorted.append(n)

            dirty_cache[n.unique_id] = is_dirty
            return is_dirty

        dfs(node)
        return topo_sorted


    def resolve_parallel(self, node: Node, use_cache: bool = True, verbose: bool = False):
        dirty_nodes = self.find_dirty_nodes_toposorted(node, use_cache)
        if not dirty_nodes:
            if obj := self._retrieve_from_cache(node, use_cache, verbose):
                return obj
            dirty_nodes.append(node)
        dirty_ids = {n.unique_id for n in dirty_nodes}

        def make_wrapper(n, resolved_deps):
            def wrapper():
                deps = [f.result() if isinstance(f, Future) else f for f in resolved_deps]
                result = _run_task(n.func, deps, n.params)
                self.cache[n.unique_id] = result
                if use_cache and n.cacheable and n.serializer:
                    n.serializer(result, n.name, n.unique_id)
                return result
            return wrapper

        futures: dict[str, Future] = {}
        with ThreadPoolExecutor() as executor:
            for n in dirty_nodes:
                resolved_deps = []
                for dep in n.dependencies:
                    if dep.is_file:
                        resolved_deps.append(dep.fpath)
                    elif dep.unique_id not in dirty_ids:
                        obj = self._retrieve_from_cache(dep, use_cache, verbose)
                        resolved_deps.append(obj)
                    else:
                        resolved_deps.append(futures[dep.unique_id])

                futures[n.unique_id] = executor.submit(make_wrapper(n, resolved_deps))

            final_result = futures[node.unique_id].result()
            return final_result


    def make(self, name, verbose: bool = False, use_cache: bool = True, parallel: bool = False):
        node   = self.build_graph(name, verbose)
        if not parallel:
            result = self.resolve(node, use_cache, verbose)
        else:
            result = self.resolve_parallel(node, use_cache)
        return result
