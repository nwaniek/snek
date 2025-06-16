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

__version__ = "0.1.6"

import sys
import configparser
import argparse
import types
import copy
from pathlib import Path
from dataclasses import fields, is_dataclass, MISSING, Field
from typing import TypeVar, Any, Type, List, Literal, Union, Tuple, get_origin, get_args, Callable

if sys.version_info < (3, 10):
    UnionType = None
else:
    from types import UnionType


import hashlib
try:
    import orjson
    def __dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode()
except ImportError:
    import json
    def __dumps(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True)

import numpy as np


def make_json_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    else:
        return obj


T = TypeVar('T')

_FIELDS = '__dataclass_fields__'

# Define missing types for compatibility
if sys.version_info < (3, 10):
    NoneType = type(None)
    EllipsisType = type(Ellipsis)
    NotImplementedType = type(NotImplemented)
else:
    NoneType = types.NoneType
    EllipsisType = types.EllipsisType
    NotImplementedType = types.NotImplementedType

_ATOMIC_TYPES = frozenset({
    # Common JSON Serializable types
    NoneType,
    bool,
    int,
    float,
    str,
    # Other common types
    complex,
    bytes,
    # Other types that are also unaffected by deepcopy
    EllipsisType,
    NotImplementedType,
    types.CodeType,
    types.BuiltinFunctionType,
    types.FunctionType,
    type,
    range,
    property,
})


def __asdict(obj: Any, field_filter: Callable[[Field], bool]) -> Any:
    obj_type = type(obj)

    if obj_type in _ATOMIC_TYPES:
        return obj

    elif is_dataclass(obj) or hasattr(obj_type, _FIELDS):
        return {
                f.name: __asdict(getattr(obj, f.name), field_filter)
                for f in fields(obj)
                if field_filter(f)
        }

    elif obj_type is list:
        return [__asdict(v, field_filter) for v in obj]

    elif obj_type is dict:
        return {__asdict(k, field_filter): __asdict(v, field_filter) for k, v in obj.items()}

    elif obj_type is tuple:
        return tuple([__asdict(v, field_filter) for v in obj])

    elif issubclass(obj_type, tuple):
        # named tuple?
        if hasattr(obj, '_fields'):
            return obj_type(*[__asdict(v, field_filter) for v in obj])
        else:
            return obj_type(__asdict(v, field_filter) for v in obj)

    elif issubclass(obj_type, dict):
        if hasattr(obj_type, 'default_factory'):
            result = obj_type(obj.default_factory)
            for k, v in obj.items():
                result[__asdict(k, field_filter)] = __asdict(v, field_filter)
            return result
        return obj_type((__asdict(k, field_filter), __asdict(v, field_filter))
                        for k, v in obj.items())

    elif issubclass(obj_type, list):
        # Assume we can create an object of this type by passing in a
        # generator
        return obj_type(__asdict(v, field_filter) for v in obj)

    else:
        return copy.deepcopy(obj)


def dataclass_to_hash(obj: object, method: str = "sha1", field_filter: Callable[[Field], bool] = lambda _: True) -> str:
    if not is_dataclass(obj):
        raise TypeError("compute_hash can only be used with dataclass instances")

    # Convert to a serializable dictionary
    config_dict = __asdict(obj, field_filter)
    config_dict = make_json_serializable(config_dict)

    # Convert to a JSON string with sorted keys for consistency
    config_json = __dumps(config_dict)

    # Compute the hash
    hash_func = getattr(hashlib, method)
    return hash_func(config_json.encode("utf-8")).hexdigest()


def dataclass_to_npz(instance: Any, filename: Path):
    """
    Saves a dataclass instance to a .npz file.

    Supports:
    - Scalars (int, float, str)
    - NumPy arrays
    - Lists of NumPy arrays
    - Dictionaries of NumPy arrays

    Parameters:
        instance (dataclass): Instance of a dataclass to save.
        filename (Path): Path to save the file.
    """
    data_dict = {}

    for field in fields(instance):
        value = getattr(instance, field.name)

        if value is None:
            data_dict[f"{field.name}_is_none"] = True
            continue

        data_dict[f"{field.name}_is_none"] = False

        if isinstance(value, list) and all(isinstance(x, np.ndarray) for x in value):
            # Handle list of NumPy arrays
            data_dict[f"{field.name}_len"] = len(value)
            for i, arr in enumerate(value):
                data_dict[f"{field.name}_{i}"] = arr

        elif isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, np.ndarray) for k, v in value.items()):
            # Handle dictionary of NumPy arrays
            data_dict[f"{field.name}_keys"] = np.array(list(value.keys()), dtype="U")  # Save keys as an array
            for key, arr in value.items():
                data_dict[f"{field.name}_{key}"] = arr

        else:
            # Scalars or direct NumPy arrays
            data_dict[field.name] = np.array(value) if isinstance(value, (int, str, float)) else value

    np.savez_compressed(filename, **data_dict)


def npz_to_dataclass(cls: Type[T], filename: Path) -> T:
    """
    Loads a dataclass instance from a .npz file.

    Supports:
    - Scalars (int, float, str)
    - NumPy arrays
    - Lists of NumPy arrays
    - Dictionaries of NumPy arrays

    Parameters:
        cls (Type[T]): The dataclass type to instantiate.
        filename (Path): Path to the saved file.

    Returns:
        T: Instance of the dataclass.
    """
    npzfile = np.load(filename, allow_pickle=False)
    init_args = {}

    for field in fields(cls):
        if npzfile.get(f"{field.name}_is_none", False):
            init_args[field.name] = None
            continue

        if f"{field.name}_len" in npzfile:
            # Load list of NumPy arrays
            length = int(npzfile[f"{field.name}_len"])
            init_args[field.name] = [npzfile[f"{field.name}_{i}"] for i in range(length)]

        elif f"{field.name}_keys" in npzfile:
            # Load dictionary of NumPy arrays
            keys = npzfile[f"{field.name}_keys"]
            init_args[field.name] = {key: npzfile[f"{field.name}_{key}"] for key in keys}

        else:
            # Load scalars or direct NumPy arrays
            init_args[field.name] = npzfile[field.name].item() if field.type in {int, str, float} else npzfile[field.name]

    return cls(**init_args)


def try_cast_value(value: str, to_type) -> tuple[Any, bool]:
    try:
        if to_type == bool:
            return value.lower() in ("1", "true", "yes", "on"), True
        if to_type == int:
            return int(value), True
        if to_type == float:
            return float(value), True
        if to_type == str:
            return value, True
        if to_type is type(None):
            return None, value.lower() in ("none", "", "null")
    except Exception:
        return None, False

    return None, False


def cast_value(value: str, to_type):
    origin = get_origin(to_type)
    args = get_args(to_type)

    if origin is Union or (UnionType is not None and origin is UnionType):
        for arg in args:
            casted, success = try_cast_value(value, arg)
            if success:
                return casted
        raise ValueError(f"Cannot cast '{value}' to any type in {to_type}")

    if origin in (list, List):
        inner_type = args[0] if args else str
        return [cast_value(v.strip(), inner_type) for v in value.split(",")]

    casted, success = try_cast_value(value, to_type)
    if success:
        return casted

    raise ValueError(f"Cannot cast '{value}' to {to_type}")


def ini_to_dataclass(ini: configparser.ConfigParser, cls, section=None):
    kwargs = {}
    for field in fields(cls):
        key = field.name
        field_type = field.type

        if is_dataclass(field_type):
            nested_section = key if section is None else f"{section}.{key}"
            nested = ini_to_dataclass(ini, field_type, section=nested_section)
            kwargs[key] = nested
            continue

        ini_section = section or "general"
        if ini_section in ini and key in ini[ini_section]:
            raw_value = ini[ini_section][key]
            value = cast_value(raw_value, field_type)
            kwargs[key] = value
        else:
            if field.default is not MISSING:
                kwargs[key] = field.default
            elif field.default_factory is not MISSING:  # type: ignore
                kwargs[key] = field.default_factory()  # type: ignore
            else:
                kwargs[key] = None

    return cls(**kwargs)


def namespace_to_obj(ns: argparse.Namespace, obj, prefix=""):
    """Fill an argparse namespace into a dataclass instance"""
    for field in fields(obj):
        key = f"{prefix}{field.name}"
        field_type = field.type

        if is_dataclass(field_type):
            namespace_to_obj(ns, getattr(obj, field.name), prefix=f"{key}_")
            continue

        cli_value = getattr(ns, key, MISSING)
        if cli_value is not MISSING and cli_value is not None:
            setattr(obj, field.name, cli_value)
    return obj


def _write_dataclass_to_ini(config_obj, ini, section_name):
    """Recursively writes a dataclass into a configparser instance (ini)."""
    if is_dataclass(config_obj):
        for field in fields(config_obj):
            key = field.name
            value = getattr(config_obj, key)

            # If the field is a dataclass itself, create a new section for it
            if is_dataclass(value):
                new_section_name = key  # section name is the field name
                if new_section_name not in ini:
                    ini.add_section(new_section_name)
                _write_dataclass_to_ini(value, ini, new_section_name)
            else:
                # Handle lists: store them as comma-separated strings
                if isinstance(value, list):
                    value = ', '.join(map(str, value))

                # Add the key-value pair to the appropriate section
                ini.set(section_name, key, str(value))


def dataclass_to_ini(config_obj, filename):
    """Converts a dataclass instance to an .ini file."""
    # Create a ConfigParser instance
    ini = configparser.ConfigParser()

    # Add a general section for the top-level Configuration fields
    ini.add_section("general")
    _write_dataclass_to_ini(config_obj, ini, "general")

    # Save the ini content to the file
    with open(filename, 'w') as configfile:
        ini.write(configfile)


def pprint_dataclass(obj, indent=2):
    print(type(obj).__name__)
    for field in fields(obj):
        value = getattr(obj, field.name)
        prefix = " " * indent
        if is_dataclass(value):
            print(f"{prefix}{field.name}: ", end="")
            pprint_dataclass(value, indent=indent + 2)
        else:
            print(f"{prefix}{field.name} ({field.type}):  {value}")


def choose_union_type(args):
    """Pick the most appropriate type from a Union, for CLI parsing."""
    # Prefer simple types for parsing
    for preferred in (str, Path, int, float, bool):
        if preferred in args:
            return preferred
    return str  # Fallback


def str_to_tuple(string: str, tuple_type: type) -> Tuple:
    value_types = get_args(tuple_type)
    values = [s.strip() for s in string.split(",")]

    if len(values) != len(value_types):
        raise ValueError("Too many or too few values in tuple")

    result = [None] * len(value_types)
    for i in range(len(value_types)):
        etype = value_types[i]
        if type(etype) is types.UnionType or get_origin(etype) is Union:
            possible_types = get_args(etype)

            success = False
            for ptype in possible_types:
                # try to convert to possible values
                try:
                    if ptype == types.NoneType:
                        if values[i].lower() == 'none':
                            result[i] = None
                            success = True
                    else:
                        result[i] = ptype(values[i])
                        success = True
                except (TypeError, ValueError):
                    pass

                if success:
                    break

            if not success:
                raise ValueError(f"Could not convert {values[i]} to {etype}")
        else:
            result[i] = etype(values[i])

    return tuple(result)


def is_union(tp):
    return get_origin(tp) in {Union, UnionType}


def dataclass_to_parser(cls, parser, prefix=""):
    for field in fields(cls):
        field_type = field.type
        origin = get_origin(field_type)
        args = get_args(field_type)

        field_name = field.name
        arg_name = f"--{prefix}{field_name.replace('_', '-')}"

        if is_dataclass(field_type):
             # Create a sub-group for nested dataclasses
             sub_group = parser.add_argument_group(f"{prefix}{field_name.capitalize()} Options")
             dataclass_to_parser(field_type, sub_group, prefix=f"{prefix}{field_name}-")
             continue

        default = argparse.SUPPRESS
        if field_type is Any:
            # default = None
            parser.add_argument(arg_name, type=None, default=default, required=False)

        elif origin is Literal:
            # default = field.default
            parser.add_argument(arg_name, choices=args, default=default, required=False)

        elif origin is list:
            contained_type = args[0]
            # default = field.default
            parser.add_argument(arg_name, nargs="*", type=contained_type, default=default, required=False)

        elif origin is tuple:
            # default = field.default
            parser.add_argument(arg_name, type=lambda x, tup_type=field_type: str_to_tuple(x, tup_type), default=default, required=False)

        elif field_type == bool:
            # default = field.default
            parser.add_argument(arg_name, type=lambda x: (str(x).lower() == 'true'), default=default, required=False)

        elif field_type == Path:
            parser.add_argument(arg_name, type=str, default=default, required=False)

        elif is_union(field_type):
            # Handle Union[str, Path], Union[str, List[str]], etc. and if it's
            # Optional[T], reduce to T
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                actual_type = non_none_args[0]
                parser.add_argument(arg_name, type=actual_type, default=default, required=False)
            else:
                selected_type = choose_union_type(non_none_args)
                parser.add_argument(arg_name, type=selected_type, default=default, required=False)

        else:
            # default = field.default
            parser.add_argument(arg_name, type=field_type, default=default, required=False)

    return parser
