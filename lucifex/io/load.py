from typing import Protocol, Any, TypeVar, Generic, Iterable, get_args
from typing_extensions import Self
from types import EllipsisType, UnionType
import pickle as pkl

import numpy as np
from matplotlib.figure import Figure
from mpi4py import MPI
from dolfinx.mesh import Mesh
from dolfinx.io import XDMFFile

from ..utils.py_utils import StrSlice, classproperty, optional_lru_cache
from ..fdm import FunctionSeries, ConstantSeries, GridSeries, NumericSeries
from .read import read
from .utils import file_path_ext


T = TypeVar('T')
class LoadObject(Generic[T], Protocol):
    """
    Callback protocol for `load` functions
    """
    _registry: dict[type, Self] = {}
    
    def __call__(
        self, 
        name: str,
        dir_path: str,
        file_name: str,
        *args,
        **kwargs,
    ) -> T:
        ...

    @classmethod
    def register(
        cls,
        load_func: Self,
        *types: type,
        strict: bool = True,
    ) -> None:
        rtrn = load_func.__annotations__.get('return')
        if strict and rtrn is None:
            raise TypeError('Function must have a return type annotation.')
        if rtrn is not None:
            cls._registry[rtrn] = load_func

        for t in types:
            cls._registry[t] = load_func

    @classmethod
    def load_object(
        cls,
        t: type | UnionType
    ) -> Self:
        try:
            return cls._registry[t]
        except KeyError:
            if type(t) is UnionType:
                for union_arg in get_args(t):
                    try:
                        return cls.load_object(union_arg)
                    except KeyError:
                        pass
            raise KeyError
        
    @classproperty
    def registry(cls):
        return cls._registry
                

@optional_lru_cache
def load_mesh(
    name: str,
    dir_path: str,
    file_name: str,
    comm: MPI.Comm | str = 'COMM_WORLD',
) -> Mesh:
    """
    NOTE the argument `comm: MPI.Comm` is not hashable so cannot be used 
    in a cached function call, but `comm: str` is hashable and so provides a workaround.
    """
    
    if isinstance(comm, str):
        comm = getattr(MPI, comm)
    
    file_path = file_path_ext(dir_path, file_name, 'xdmf')

    with XDMFFile(comm, file_path, "r") as xdmf:
        msh = xdmf.read_mesh(name=name)

    return msh


@optional_lru_cache
def load_function_series(
    name: str,
    dir_path: str,
    file_name: str,
    mesh: str | Mesh,
    *elem_slc: tuple[str, int] | tuple[str, int, int] | StrSlice,
) -> FunctionSeries:
    
    if isinstance(mesh, str):
        mesh = load_mesh(mesh, dir_path, file_name)

    match elem_slc:
        case (elem, slc) if isinstance(elem, tuple):
            elem, slc = elem_slc
        case (slc, elem) if isinstance(elem, tuple):
            slc, elem = elem_slc
        case (elem, ) if isinstance(elem, tuple):
            elem, = elem_slc
            slc = ':'
        case (slc, ):
            slc, = elem_slc
            elem = ('P', 1)
        case ():
            slc = ':'
            elem = ('P', 1)
            
    f = FunctionSeries((mesh, *elem), name) 
    read(f, dir_path, file_name, slc)
    return f


@optional_lru_cache
def load_constant_series(
    name: str,
    dir_path: str,
    file_name: str,
    mesh: str | tuple[str, str] | Mesh,
    *shape_slc: tuple[int, ...] | StrSlice,
) -> ConstantSeries:

    match shape_slc:
        case (shape, slc) if isinstance(shape, tuple):
            shape, slc = shape_slc
        case (slc, shape) if isinstance(shape, tuple):
            slc, shape = shape_slc
        case (shape, ) if isinstance(shape, tuple):
            shape, = shape_slc
            slc = ':'
        case (slc, ):
            slc, = shape_slc
            shape = ()
        case ():
            shape = ()
            slc = ':'

    if isinstance(mesh, str):
        mesh = load_mesh(mesh, dir_path, file_name)
    if isinstance(mesh, tuple):
        mesh, mesh_file_name = mesh
        mesh = load_mesh(mesh, dir_path, mesh_file_name)

    c = ConstantSeries(mesh, name, shape=shape)
    read(c, dir_path, file_name, slc)
    return c


@optional_lru_cache
def load_grid_series(
    name: str,
    dir_path: str,
    file_name: str,
    sep: str = '__'
) -> GridSeries:
    try:
        return load_npz_dict(dir_path, file_name, sep, search=name)[name]
    except KeyError:
        raise ValueError(f"'{name}' not found in {file_name}.")


@optional_lru_cache
def load_numeric_series(
    name: str,
    dir_path: str,
    file_name: str,
    sep: str = '__'
) -> NumericSeries:
    try:
        return load_npz_dict(dir_path, file_name, sep, search=name)[name]
    except KeyError:
        raise ValueError(f"'{name}' not found in {file_name}.")


@optional_lru_cache
def load_txt_dict(
    dir_path: str,
    file_name: str,
    eval_locals: Iterable[type] = (),
    sep: str = ' = ',
    skip: Iterable[int | str] = (),
) -> dict[str, Any | EllipsisType]:

    file_path = file_path_ext(dir_path, file_name, 'txt')
    params = {}

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i in skip:
                continue
            name, val_str = line.split(sep)
            if name in skip:
                continue
            val_str = val_str.strip() 
            try:
                val = eval(val_str, globals(), {i.__name__: i for i in eval_locals})
                params[name] = val
            except Exception as ex:
                params[name] = type(ex).__name__

    return params


@optional_lru_cache
def load_npz_dict(
    dir_path: str,
    file_name: str,
    sep: str = '__',
    *,
    search: str | None = None,
) -> dict[str, float | np.ndarray | NumericSeries | GridSeries]:
    file_path = file_path_ext(dir_path, file_name, 'npz')
    npz_dict: dict[str, np.ndarray] = np.load(file_path)

    d: dict[str, float | np.ndarray | dict[str, float] | dict[str | float, float | np.ndarray]] = {}
    for k, v in npz_dict.items():
        if search is not None and not k.startswith(search):
            continue
        if isinstance(v, np.ndarray) and v.shape == ():
            v = v[()]
        match k.split(sep):
            case n, axis_or_time: 
                if n not in d:
                    d[n] = {}
                try:
                    d[n].update({float(axis_or_time): v}) 
                except ValueError:
                    d[n].update({axis_or_time: v}) 
            case _:
                d[k] = v 

    rtrn: dict[str, float | np.ndarray | NumericSeries | GridSeries] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            series = {i: j for i, j in v.items() if isinstance(i, float)}
            axes = {i: j for i, j in v.items() if isinstance(i, str)}
            if axes:
                v = GridSeries(list(series.values()), list(series.keys()), tuple(axes.values()), k)
            else:
                v = NumericSeries(list(series.values()), list(series.keys()), k)
        rtrn[k] = v

    return rtrn


def load_figure(
    dir_path: str,
    file_name: str,
) -> Figure:
    file_path = file_path_ext(dir_path, file_name, 'pickle')
    return pkl.load(file_path, 'rb')


@optional_lru_cache
def load_value(
    name: str,
    dir_path: str,
    file_name: str,
    *args,
    **kwargs,
) -> Any:
    try:
        d = load_txt_dict(dir_path, file_name, *args, **kwargs)
    except FileNotFoundError:
        d = load_npz_dict(dir_path, file_name, *args, **kwargs, search=name)
    if name in d:
        return d[name]
    raise  ValueError(f"'{name}' not found in {file_name}.")
    

LoadObject.register(load_mesh)
LoadObject.register(load_function_series)
LoadObject.register(load_constant_series)
LoadObject.register(load_grid_series)
LoadObject.register(load_numeric_series)
LoadObject.register(load_value, str, int, float, np.ndarray)

T = TypeVar('T')
def load(
    t: type[T], 
    use_cache: bool = True,
    clear_cache: bool = False,
) -> LoadObject[T]:
    load_obj = LoadObject.load_object(t)
    try:
        return load_obj(use_cache=use_cache, clear_cache=clear_cache)
    except TypeError:
        return load_obj   



# def load_parameter(
#     name: str,
#     dir_path: str,
#     file_name: str,
#     eval_locals: Iterable[type] = (),
#     sep: str = ' = ',
#     skip: Iterable[int | str] = (),
# ) -> Any | EllipsisType:
#     return load_txt_dict(dir_path, file_name, eval_locals, sep, skip)[name]