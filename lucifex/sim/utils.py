import csv
from inspect import signature
from collections.abc import Iterable
from typing import Callable

from ..io.utils import file_path_ext


def arg_name_collisions(
    *callables: Iterable[Callable],
) -> list[str]:
    names = list(signature(callables[0]).parameters)
    collisions = []

    for c in callables[1:]:
        _names = tuple(signature(c).parameters)
        for n in _names:
            if n in names:
                collisions.append(n)
        names.extend(_names)

    return collisions
   

class ArgNameCollisionError(RuntimeError):
    def __init__(self, collisions: list[str]):
        msg = 'Argument names must be unique to each callable'
        if len(collisions) == 1:
            msg = f"{msg}, but '{collisions[0]}' is shared"
        if len(collisions) > 1:
            msg = f"{msg}, but {tuple(collisions)} are shared"
        super().__init__(msg)


def write_texec(
    texec_log: dict[str, list[float]],  
    dir_path: str,
    file_name: str,
    
):
    file_path = file_path_ext(dir_path, file_name, 'csv')
    n_rows = max(len(i) for i in texec_log.values())
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(texec_log.keys())
        for i in range(n_rows):
            row = []
            for v in texec_log.values():
                try:
                    row.append(v[i])
                except IndexError:
                    row.append('')
            writer.writerow(row)


def load_texec(
    dir_path: str,
    file_name: str,
) -> dict[str, list[float]]:
    file_path = file_path_ext(dir_path, file_name, 'csv', mkdir=False)
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                keys = row
                texec_log = {k: list() for k in keys}
            else:
                [texec_log[k].append(float(i)) for k, i in zip(keys, row, strict=True) if i != '']

    return texec_log

