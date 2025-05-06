#!/usr/bin/env python

from pathlib import Path
import numpy as np


def flatten_dict(d: dict, parent_key: str = '', sep: str = '/') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            nested = flatten_dict(v, new_key, sep=sep)
            items.extend(nested.items())

        elif isinstance(v, list):
            items.append((f"{new_key}::len", len(v)))
            for i, item in enumerate(v):

                if isinstance(item, dict):
                    nested = flatten_dict({str(i): item}, new_key, sep=sep)
                    items.extend(nested.items())
                else:
                    items.extend([(f"{new_key}::{i}", item)])
        else:
            items.append((new_key, v))
    return dict(items)


def dict_to_npz(d: dict, filename: Path):
    flat_dict = flatten_dict(d)
    npz_dict = {}
    for key, value in flat_dict.items():
        if isinstance(value, (int, float, str)):
            npz_dict[key] = np.array(value)
        elif isinstance(value, np.ndarray):
            npz_dict[key] = value
        else:
            raise ValueError(f"Unsupported type {type(value)} for key {key}")

    np.savez_compressed(filename, **npz_dict)


def reconstruct_lists(d):
    keys_to_delete = []

    for k, v in list(d.items()):
        if isinstance(v, dict):
            reconstruct_lists(v)

        if isinstance(k, str) and k.endswith('::len'):
            base = k[:-6]
            length = int(v)
            reconstructed = []
            for i in range(length):
                item_key = f"{base}::{i}"
                if item_key in d:
                    reconstructed.append(d[item_key])
                    keys_to_delete.append(item_key)
            d[base] = reconstructed
            keys_to_delete.append(k)

    for k in keys_to_delete:
        del d[k]


def unflatten_dict(flat_dict: dict, sep: str = '/') -> dict:
    tree = {}

    for compound_key, value in flat_dict.items():
        path = compound_key.split(sep)
        d = tree
        for key in path[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[path[-1]] = value

    def reconstruct_lists(d):
        if not isinstance(d, dict):
            return

        keys = list(d.keys())
        for key in keys:
            value = d[key]
            if isinstance(value, dict):
                reconstruct_lists(value)

        for key in keys:
            if key.endswith('::len'):
                base = key[:-5]
                length = int(d[key])
                reconstructed = []
                for i in range(length):
                    item_key = f"{base}::{i}"
                    if item_key in d:
                        reconstructed.append(d[item_key])
                # Replace base key with reconstructed list
                d[base] = reconstructed
                # Cleanup
                d.pop(key)
                for i in range(length):
                    d.pop(f"{base}::{i}", None)

    reconstruct_lists(tree)
    return tree


def npz_to_dict(filename: Path) -> dict:
    loaded = np.load(filename, allow_pickle=False)
    flat = {k: loaded[k].item() if loaded[k].shape == () else loaded[k] for k in loaded}
    return unflatten_dict(flat)
