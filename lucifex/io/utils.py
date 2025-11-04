import datetime
import os
from typing import Any
from collections.abc import Iterable

from dolfinx.fem import Function, FunctionSpace
from ..fdm import FunctionSeries
from ..utils.fem_utils import ScalarVectorError, is_discontinuous_lagrange


def create_dir_path(
    namespace: dict[str, Any],
    *,
    dir_base: str,
    dir_params: Iterable[str] | str,
    dir_prefix: str | None,
    dir_suffix: str | None,
    dir_timestamp: bool | None,
    dir_seps: tuple[str, str] = ('|', '__'),
    mkdir: bool = False,
) -> str:
    
    dir_name = ''
    arg_sep, sep = dir_seps

    if isinstance(dir_params, str):
        dir_params = dir_params.split()
    dir_params_values = arg_sep.join([f"{arg_name}={namespace[arg_name]}" for arg_name in dir_params])
    dir_name = f'{dir_name}{dir_params_values}'

    if dir_prefix:
        dir_name = f"{dir_prefix}{sep}{dir_name}"
    
    if dir_suffix:
        dir_name = f"{dir_name}{sep}{dir_suffix}"

    if dir_timestamp:
        time = str(datetime.datetime.now())
        whitespace_delims = " ", "_"
        year_month_day_delims = "-", "-"
        hour_min_sec_delims = ":", "-"
        decimal_delims = ".", "."
        time = (
            time.replace(*whitespace_delims)
            .replace(*year_month_day_delims)
            .replace(*hour_min_sec_delims)
            .replace(*decimal_delims)
        )
        dir_name = f"{dir_name}{sep}{time}"

    if mkdir:
        os.makedirs(dir_name, exist_ok=True)

    return os.path.join(dir_base, dir_name)


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
        

def get_ipynb_file_name(
    key: str = '__vsc_ipynb_file__',
    ext: bool = False,
) -> str:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            _globals = ip.user_global_ns 
    except Exception:
        _globals = globals()

    basename = os.path.basename(_globals[key])

    if ext:
        return basename
    else:
        return os.path.splitext(basename)[0]


def io_element(
    u: FunctionSpace | Function | FunctionSeries,
) -> tuple[str, int] | tuple[str, int, int]:
    if isinstance(u, FunctionSpace):
        fs = u
    else:
        fs = u.function_space

    if is_discontinuous_lagrange(fs, 0):
        elem = ('DP', 0)
    else:
        elem = ('P', 1)
    match u.shape:
        case ():
            return elem
        case (dim, ):
            return (*elem, dim)
        case _:
            raise ScalarVectorError(u)