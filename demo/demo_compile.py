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


def make_obj(dm, source, obj_file=None):
    obj_file = Path(source).with_suffix('.o') if not obj_file else obj_file

    @dm.target(provides=str(obj_file), requires=["@" + str(source)], cacheable=False)
    def _obj(c_file) -> Path:
        if not compile_c_file(c_file, obj_file):
            raise RuntimeError(f"Compilation failed for {c_file}")
        return obj_file


def link(dm, target, objs):
    print(target)
    print(objs)

    @dm.target(provides=target, requires=objs, cacheable=False)
    def _binary(*args) -> Path:
        args = [Path(f) if not isinstance(f, Path) else f for f in args]
        target_file = Path(target) if not isinstance(target, Path) else target
        if not link_object_files(args, target_file):
            print("Linking failed")
        return target_file


dm = DependencyManager()

@dm.target(requires=['demo'], cacheable=False)
def run(demo):
    cmd = ['./' + str(demo)]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return 'ERROR'


make_obj(dm, 'demo.c')
make_obj(dm, 'demo2.c')
link(dm, 'demo', ['demo.o', 'demo2.o'])

print(dm.make('run', use_cache=False))


