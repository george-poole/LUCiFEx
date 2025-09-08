from collections.abc import Iterable
from typing import overload

from ..utils import is_structured
from ..fdm.series import ConstantSeries, FunctionSeries, GridSeries, NumericSeries
from ..io import write, load_mesh, load_function_series, load_constant_series

from .simulation import Simulation


@overload
def postprocess_structured(
    load_mesh_args: tuple[str, str, str],
    grid_file_name: str | None = None,
    numeric_file_name: str | None = None,
    *,
    grids: Iterable[tuple[str, str, tuple]] = (),
    numerics: Iterable[tuple[str, str, tuple]] = (),
) -> None:
    ...

@overload
def postprocess_structured(
    sim: Simulation,
    grid_file_name: str | None = None,
    numeric_file_name: str | None = None,
    *,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
) -> None:
    ...


def postprocess_structured(
    arg,
    grid_file_name: str | None = None,
    numeric_file_name: str | None = None,
    **kwargs,
) -> None:
    
    if grid_file_name is None:
        grid_file_name = GridSeries.__name__
    if numeric_file_name is None:
        numeric_file_name = NumericSeries.__name__

    if isinstance(arg, Simulation):
        EXCL, INCL = 'exclude', 'include'
        _kwargs = {EXCL: (), INCL: ()}
        _kwargs.update(kwargs)
        exclude = _kwargs[EXCL]
        include = _kwargs[INCL]
        
        grid_series = [i.series for i in arg.solvers if isinstance(i.series, FunctionSeries)]
        numeric_series = [i.series for i in arg.solvers if isinstance(i.series, ConstantSeries)]
        
        assert (exclude, include).count(()) != 0
        if exclude:
            condition = lambda n: n not in exclude
        else:
            condition = lambda n: n in include

        elem = ('P', 1) #FIXME safe to assume P1? DP0 in some cases?
        grid_series = [(i.name, arg.write_file[i.name], elem) for i in grid_series if condition(i.name)]
        numeric_series = [(i.name, arg.write_file[i.name], i.shape) for i in numeric_series if condition(i.name)]
        dir_path = arg.dir_path
        mesh = arg.t.mesh
    else:
        dir_path, mesh_name, mesh_file_name = arg
        GRDS, NUMS = 'grids', 'numerics'
        _kwargs = {GRDS: (), NUMS: ()}
        _kwargs.update(kwargs)
        grid_series = _kwargs[GRDS]
        numeric_series = _kwargs[NUMS]
        mesh = load_mesh(mesh_name, dir_path, mesh_file_name)

    if not is_structured(mesh):
        raise ValueError

    for i in grid_series:
        name, file_name = i[:2]
        elem = i[2:]
        uxt_load = load_function_series(name, dir_path, file_name, mesh, *elem)
        write(
            GridSeries.from_series(uxt_load), 
            file_name=grid_file_name, 
            dir_path=dir_path,
        )
        
    for i in numeric_series:
        name, file_name = i[:2]
        shape = i[2:]
        uxt_load = load_constant_series(name, dir_path, file_name, mesh, *shape)
        write(
            NumericSeries.from_series(uxt_load), 
            file_name=numeric_file_name, 
            dir_path=dir_path,
        )