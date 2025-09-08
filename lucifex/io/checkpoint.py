import os
import glob
from typing import Iterable

from ..fdm.series import FunctionSeries, ConstantSeries
from .write import write, clear_write_cache
from .read import read


def write_checkpoint(
    series: list[FunctionSeries | ConstantSeries],
    t: ConstantSeries,
    file_name: str,
    dir_path: str,
) -> None:
    for u in series:
        order = min(t.order, u.order)
        for n in range(order):
            write(u[-n], file_name, dir_path, t[-n], mode='c')


def read_checkpoint(
    series: list[FunctionSeries | ConstantSeries],
    dir_path: str,
    file_name: str,
) -> None:
    [read(s, file_name, dir_path, slice(-s.order, -1)) for s in series] 


def reset_directory(
    dir_path,
    remove: Iterable[str],
    keep: Iterable[str] = (),
) -> None:
    remove_paths = []
    for pattern in remove:
        remove_paths.extend(glob.glob(f'{dir_path}/{pattern}'))

    keep_paths = []
    for pattern in keep:
        keep_paths.extend(glob.glob(f'{dir_path}/{pattern}'))

    remove_paths = set(remove_paths) - set(keep_paths)
    [os.remove(f) for f in remove_paths]
    clear_write_cache()



