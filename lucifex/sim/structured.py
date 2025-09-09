import os
from collections.abc import Iterable
from typing import overload

from ..utils import is_structured
from ..utils.fem_utils import ScalarVectorError, is_discontinuous_lagrange
from ..fdm.series import ConstantSeries, FunctionSeries, GridSeries, NumericSeries
from ..io import write, load_mesh, load_function_series, load_constant_series

from .simulation import Simulation
from .utils import file_path_ext


@overload
def write_structured(
    load_mesh_args: tuple[str, str, str],
    grid_file_name: str | None = None,
    numeric_file_name: str | None = None,
    *,
    load_grid_args: Iterable[tuple[str, str, tuple]] = (),
    load_numeric_args: Iterable[tuple[str, str, tuple]] = (),
    delete_h5_xdmf: bool = False,
) -> None:
    ...

@overload
def write_structured(
    sim: Simulation,
    grid_file_name: str | None = None,
    numeric_file_name: str | None = None,
    *,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
    delete_h5_xdmf: bool = False,
) -> None:
    ...


def write_structured(
    arg,
    grid_file_name: str | None = None,
    numeric_file_name: str | None = None,
    **kwargs,
) -> None:
    
    if grid_file_name is None:
        grid_file_name = GridSeries.__name__
    if numeric_file_name is None:
        numeric_file_name = NumericSeries.__name__

    DEL = 'delete_h5_xdmf'
    _kwargs = {DEL: False}

    if isinstance(arg, Simulation):
        EXCL, INCL = 'exclude', 'include'
        _kwargs.update({EXCL: (), INCL: ()})
        _kwargs.update(kwargs)
        exclude = _kwargs[EXCL]
        include = _kwargs[INCL]
        
        grid_series = [i.series for i in arg.solvers if isinstance(i.series, FunctionSeries)]
        numeric_series = [i.series for i in arg.solvers if isinstance(i.series, ConstantSeries)]
        
        if include:
            _include = lambda n: n in include and not n in exclude
        else:
            _include = lambda n: n not in exclude

        load_grid_args = [(i.name, arg.write_file[i.name], elem_io(i)) for i in grid_series if _include(i.name)]
        load_numeric_args = [(i.name, arg.write_file[i.name], i.shape) for i in numeric_series if _include(i.name)]
        dir_path = arg.dir_path
        mesh = arg.t.mesh
    else:
        dir_path, mesh_name, mesh_file_name = arg
        GRDS, NUMS = 'grids', 'numerics'
        _kwargs.update({GRDS: (), NUMS: ()})
        _kwargs.update(kwargs)
        load_grid_args = _kwargs[GRDS]
        load_numeric_args = _kwargs[NUMS]
        mesh = load_mesh(mesh_name, dir_path, mesh_file_name)

    if not is_structured(mesh):
        raise ValueError('Expected a structured mesh')

    for i in load_grid_args:
        name, file_name = i[:2]
        elem = i[2:]
        uxt_load = load_function_series(name, dir_path, file_name, mesh, *elem)
        write(
            GridSeries.from_series(uxt_load), 
            file_name=grid_file_name, 
            dir_path=dir_path,
        )
        
    for i in load_numeric_args:
        name, file_name = i[:2]
        shape = i[2:]
        uxt_load = load_constant_series(name, dir_path, file_name, mesh, *shape)
        write(
            NumericSeries.from_series(uxt_load), 
            file_name=numeric_file_name, 
            dir_path=dir_path,
        )

    if _kwargs[DEL]:
        file_names = set((fn for _, fn, *_ in (*load_grid_args, *load_numeric_args)))
        file_paths = [
            *(file_path_ext(dir_path, fn, 'h5', mkdir=False) for fn in file_names),
            *(file_path_ext(dir_path, fn, 'xdmf', mkdir=False) for fn in file_names),
        ]          
        for fp in file_paths:
            os.remove(fp)


def elem_io(u: FunctionSeries) -> tuple[str, int] | tuple[str, int, int]:
    if is_discontinuous_lagrange(u.function_space, 0):
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
