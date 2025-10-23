import os
from collections.abc import Iterable
from typing import overload, TypeAlias
from typing_extensions import Unpack
from functools import singledispatch

from dolfinx.mesh import Mesh

from ..utils import is_cartesian, MultipleDispatchTypeError
from ..utils.fem_utils import ScalarVectorError, is_discontinuous_lagrange
from ..fdm.series import ConstantSeries, FunctionSeries, GridSeries, NumericSeries
from ..io import write, load_mesh, load_function_series, load_constant_series

from .simulation import Simulation
from .utils import file_path_ext


ObjectName: TypeAlias = str
DirPath: TypeAlias = str
FileName: TypeAlias = str
@overload
def postprocess_grids(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    load_grid_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    load_numeric_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    delete_h5_xdmf: bool = False,
    mode: str = 'a',
    grid_file_name: str = GridSeries.__name__,
    numeric_file_name: str = NumericSeries.__name__,
) -> None:
    ...


@overload
def postprocess_grids(
    sim: Simulation,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
    delete_h5_xdmf: bool = False,
    mode: str = 'a',
    grid_file_name: str = GridSeries.__name__,
    numeric_file_name: str = NumericSeries.__name__,
) -> None:
    ...


def postprocess_grids(*args, **kwargs) -> None:
    return _postprocess_grids(*args, **kwargs)
    

@singledispatch
def _postprocess_grids(arg, *_, **__):
    raise MultipleDispatchTypeError(arg, _postprocess_grids)


@_postprocess_grids.register(Simulation)
def _(
    sim: Simulation,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
    delete_h5_xdmf: bool = False,
    mode: str = 'a',
    grid_file_name: str = GridSeries.__name__,
    numeric_file_name: str = NumericSeries.__name__,
):
    grid_series = [i.series for i in sim.problems if isinstance(i.series, FunctionSeries)]
    numeric_series = [i.series for i in sim.problems if isinstance(i.series, ConstantSeries)]
    
    if include:
        _include = lambda n: n in include and not n in exclude
    else:
        _include = lambda n: n not in exclude

    dir_path = sim.dir_path
    mesh = sim.t.mesh
    load_grid_args = [(i.name, sim.write_file[i.name], elem_io(i)) for i in grid_series if _include(i.name)]
    load_numeric_args = [(i.name, sim.write_file[i.name], i.shape) for i in numeric_series if _include(i.name)]

    return _postprocess_grids(
        dir_path, 
        mesh, 
        load_grid_args, 
        load_numeric_args, 
        delete_h5_xdmf, 
        mode,
        grid_file_name, 
        numeric_file_name,
    )


@_postprocess_grids.register(str)
def _(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    load_grid_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    load_numeric_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    delete_h5_xdmf: bool = False,
    mode: str = 'a',
    grid_file_name: str = GridSeries.__name__,
    numeric_file_name: str = NumericSeries.__name__,
):
    if not isinstance(mesh, Mesh):
        mesh_name, mesh_file_name = mesh
        mesh = load_mesh(mesh_name, dir_path, mesh_file_name)
    else:
        mesh = mesh

    if not is_cartesian(mesh):
        raise ValueError('Expected a Cartesian mesh')

    for i in load_grid_args:
        name, file_name = i[:2]
        elem = i[2:]
        uxt_load = load_function_series(name, dir_path, file_name, mesh, *elem)
        write(
            GridSeries.from_series(uxt_load), 
            file_name=grid_file_name, 
            dir_path=dir_path,
            mode=mode,
        )
        
    for i in load_numeric_args:
        name, file_name = i[:2]
        shape = i[2:]
        uxt_load = load_constant_series(name, dir_path, file_name, mesh, *shape)
        write(
            NumericSeries.from_series(uxt_load), 
            file_name=numeric_file_name, 
            dir_path=dir_path,
            mode=mode,
        )

    if delete_h5_xdmf:
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