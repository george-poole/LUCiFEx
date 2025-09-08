import csv
from inspect import signature
from typing import Callable
from collections.abc import Iterable

from ..io.utils import file_path_ext


def signature_name_collision(*callables: Iterable[Callable]) -> bool:
    sig_names = list(signature(callables[0]).parameters)

    for c in callables[1:]:
        new_names = tuple(signature(c).parameters)
        for n in new_names:
            if n in sig_names:
                return True
        sig_names.extend(new_names)

    return False


def write_texec_log(
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


def load_texec_log(
    dir_path: str,
    file_name: str,
) -> dict[str, list[float]]:
    file_path = file_path_ext(dir_path, file_name, 'csv')
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                keys = row
                texec_log = {k: list() for k in keys}
            else:
                [texec_log[k].append(float(i)) for k, i in zip(keys, row, strict=True) if i != '']

    return texec_log
