#!/usr/bin/env python

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess
from snek import DependencyManager
from pathlib import Path
from typing import List


def compile_c_file(source_file: Path, output_file: Path, extra_args=None):
    if extra_args is None:
        extra_args = []
    if not (compiler := os.getenv('CC')):
        compiler = 'gcc'

    cmd = [compiler, str(source_file), "-c", "-o", str(output_file)] + extra_args
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        return False


def link_object_files(object_files: List[Path], output_file: Path, extra_args=None):
    if extra_args is None:
        extra_args = []
    if not (compiler := os.getenv('CC')):
        compiler = 'gcc'

    cmd = [compiler] + [str(obj) for obj in object_files] + ["-o", str(output_file)] + extra_args

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


dm = DependencyManager()


@dm.target(requires=["@demo.c"], cacheable=False)
def demo_o(c_file) -> Path:
    o_file = c_file.with_suffix('.o')
    compile_c_file(c_file, o_file)
    return o_file


@dm.target(requires=["demo_o"], cacheable=False)
def demo_binary(obj_file) -> Path:
    output_file = Path('demo')
    link_object_files([obj_file], output_file)
    return output_file


@dm.target(cacheable=False)
def run(demo_binary):
    result = subprocess.run(
        ["./" + str(demo_binary)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout


result = dm.make("run", use_cache=False)
print(result, end="")

