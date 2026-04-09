import os
import glob
import pickle
import hashlib
import time
import datetime
from typing import Any
from collections.abc import Iterable
from natsort import natsorted

from dolfinx.fem import Function, FunctionSpace

from ..fdm import FunctionSeries
from ..utils.fenicsx_utils import IsNotScalarOrVectorError, is_discontinuous_lagrange


def create_dir_path(
    namespace: dict[str, Any],
    *,
    dir_root: str,
    dir_params: Iterable[str] | str = (),
    dir_prefix: str | None = None,
    dir_suffix: str | None = None,
    dir_datetime: bool | slice = False,
    dir_uid: bool = False,
    dir_seps: tuple[str, str, str] = ('|', '__', '__'),
    mkdir: bool = False,
) -> str:
    
    dir_name = ''
    arg_sep, prefix_sep, suffix_sep = dir_seps

    if isinstance(dir_params, str):
        dir_params = dir_params.split()
    dir_params_values = arg_sep.join([f"{arg_name}={namespace[arg_name]}" for arg_name in dir_params])
    dir_name = ''.join((dir_name, dir_params_values))

    if dir_prefix:
        dir_name = prefix_sep.join((dir_prefix, dir_name)) if dir_name else dir_prefix
    
    if dir_suffix:
        dir_name = suffix_sep.join((dir_name, dir_suffix)) if dir_name else dir_suffix

    if dir_datetime:
        if dir_datetime is True:
            slc = slice(2, -4)
        else:
            slc = dir_datetime
        time = str(datetime.datetime.now())[slc]
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
        dir_name = suffix_sep.join((dir_name, time)) if dir_name else time

    if dir_uid:
        uid = create_uid(namespace)
        dir_name = suffix_sep.join((dir_name, uid)) if dir_name else uid

    if mkdir:
        os.makedirs(dir_name, exist_ok=True)

    return os.path.join(dir_root, dir_name)


def create_uid(
    namespace: dict[str, Any],
    digest_size: int = 8,
    ignore_keys: bool = False,
) -> str:
    d = dict(sorted(namespace.items()))
    if ignore_keys:
        obj = tuple(d.values())
    else:
        obj = d
    try:
        serialized = pickle.dumps(obj)
    except pickle.PicklingError:
        import dill
        serialized = dill.dumps(obj)
    return hashlib.blake2b(serialized, digest_size=digest_size).hexdigest()


def dir_path_exists(
    dir_path: str,
    glob_prefix: str | None = None,
    glob_suffix: str | None = None,
    unique: bool = True,
    contains: str | None = None,
) -> bool:

    if contains is not None:
        contains_file = dir_path_contains
    else:
        contains_file = lambda *_: True

    if unique and not glob_prefix and not glob_suffix:
        return os.path.exists(dir_path) and contains_file(dir_path, contains)
    else:
        if glob_prefix is None:
            glob_prefix = ''
        if glob_suffix is None:
            glob_suffix = ''
        globbed = glob.glob(f'{glob_prefix}{dir_path}{glob_suffix}')
        if unique:
            return len(globbed) == 1 and contains_file(globbed[0], contains)
        else:
            return len(globbed) >= 1 and all(contains_file(i, contains) for i in globbed)


def dir_path_is_running(
    dir_path: str,
    dt_thresh: float,
    file_name: str | Iterable[str] |  None = None,
) -> bool:
    if file_name is None:
        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

    if isinstance(file_name, str):
        file_name = [file_name]
    if file_name is not None:
        file_paths = [os.path.join(dir_path, f) for f in file_name]

    file_paths = [fp for fp in file_paths if os.path.isfile(fp)]

    if not file_paths:
        raise ValueError(f'Not files found in {dir_path}')

    t_now = time.time()
    for fp in file_paths:
        t_mod = os.path.getmtime(fp)
        if t_now - t_mod < dt_thresh:
            return True
    return False
        

def dir_path_contains(
    dir_path: str,
    file_name:  str,    
) -> bool:
    return os.path.exists(os.path.join(dir_path, file_name))
    
    
def find_dir_paths(
    dir_root: str,
    *,
    include: str | Iterable[str] = '*',
    exclude: str | Iterable[str] = (),
    contains: str | Iterable[str] | None = None,
) -> list[str]:

    dir_paths = set()

    include = [include] if isinstance(include, str) else include
    for pattern in include:
        dir_paths.update(glob.glob(f'{dir_root}/{pattern}/'))

    exclude = [exclude] if isinstance(exclude, str) else exclude
    for pattern in exclude:
        [dir_paths.discard(i) for i in glob.glob(f'{dir_root}/{pattern}/')]

    dir_paths = natsorted(dir_paths)

    if contains is not None:
        if isinstance(contains, str):
            contains = [contains]
        for name in contains:
            dir_paths = [i for i in dir_paths if dir_path_contains(i, name)]

    return dir_paths


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


def dofs_array_dim(
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
        

def xdmf_element(
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
    
    match u.ufl_shape:
        case ():
            return elem
        case (dim, ):
            return (*elem, dim)
        case _:
            raise IsNotScalarOrVectorError(u)