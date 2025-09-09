import datetime
import os
from typing import Any
from collections.abc import Iterable


def create_path(
    namespace: dict[str, Any],
    *,
    dir_base: str,
    dir_params: Iterable[str] | str,
    dir_label: str | None,
    dir_timestamp: bool | None,
    dir_seps: tuple[str, str] = ('|', '__'),
    mkdir: bool = False,
) -> str:
    
    arg_sep, tag_sep = dir_seps
    dir_path = dir_base

    if isinstance(dir_params, str):
        dir_params = dir_params.split()

    for i, arg_name in enumerate(dir_params):
        if i == 0:
            dir_path = f"{dir_path}/{arg_name}={namespace[arg_name]}"
        else:
            dir_path = f"{dir_path}{arg_sep}{arg_name}={namespace[arg_name]}"

    if dir_label:
        dir_path = f"{dir_path}{tag_sep}{dir_label}"

    if dir_timestamp:
        tstamp = str(datetime.datetime.now())
        whitespace_delims = " ", "_"
        year_month_day_delims = "-", "-"
        hour_min_sec_delims = ":", "-"
        decimal_delims = ".", "."
        tstamp = (
            tstamp.replace(*whitespace_delims)
            .replace(*year_month_day_delims)
            .replace(*hour_min_sec_delims)
            .replace(*decimal_delims)
        )
        dir_path = f"{dir_path}{tag_sep}{tstamp}"

    if mkdir:
        os.makedirs(dir_path, exist_ok=True)

    return dir_path


def file_name_ext(
    file_name: str,
    ext: str | None,
) -> str:
    if ext is None:
        return file_name
    if ext[0] != '.':
        ext = f'.{ext}'
    if file_name[-len(ext):] != ext:
        file_name = f'{file_name}{ext}'
    return file_name


def file_path_ext(
    dir_path: str | None,
    file_name: str,
    ext: str | None,
    mkdir: bool = True
) -> str:
    """
    `'dir_path/file_name.ext'`
    """
    if dir_path is None:
        dir_path = os.getcwd()
    if mkdir:
        os.makedirs(dir_path, exist_ok=True)
    file_name = file_name_ext(file_name, ext)
    file_path = os.path.join(dir_path, file_name)
    return file_path


def io_array_dim(
    shape: tuple[int, ...]
) -> int:
    match shape:
        case ():
            return 1
        case (dim, ):
            return dim
        case (dim0, dim1):
            return dim0 * dim1
        case _:
            raise NotImplementedError(f'I/O with shape {shape} is not supported.')