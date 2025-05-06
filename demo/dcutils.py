#!/usr/bin/env python

import configparser
import argparse
import types
from pathlib import Path
from dataclasses import fields, is_dataclass, MISSING, asdict
from typing import TypeVar, Any, Type, List, Literal, Union, Tuple, get_origin, get_args
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

T = TypeVar('T')


def dataclass_to_hash(obj: object, method: str = "sha1") -> str:
    if not is_dataclass(obj):
        raise TypeError("compute_hash can only be used with dataclass instances")

    config_dict = asdict(obj)
    config_json = __dumps(config_dict)
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


def cast_value(value: str, to_type):
    origin = get_origin(to_type)
    args = get_args(to_type)

    if to_type == bool:
        return value.lower() in ("1", "true", "yes", "on")
    if to_type == int:
        return int(value)
    if to_type == float:
        return float(value)
    if to_type == str:
        return value
    if origin in (list, List):
        return [cast_value(v.strip(), args[0]) for v in value.split(",")]
    return value


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

        elif origin is Union:
            # specify no type, let the class verification do it
            # TODO: make this pick a better type, maybe synthesize an
            # appropriate lambda
            parser.add_argument(arg_name, type=None, default=default, required=False)

        else:
            # default = field.default
            parser.add_argument(arg_name, type=field_type, default=default, required=False)

    return parser
