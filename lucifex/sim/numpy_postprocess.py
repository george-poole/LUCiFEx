import os
from collections.abc import Iterable
from typing import overload, TypeAlias
from typing_extensions import Unpack
from functools import singledispatch

from dolfinx.mesh import Mesh

from ..utils import is_cartesian, MultipleDispatchTypeError, CellType, UnstructuredQuadError
from ..fdm import ConstantSeries, FunctionSeries, GridSeries, NumericSeries, TriangulationSeries
from ..io import (
    write, 
    load_mesh, 
    load_function_series, 
    load_constant_series,
)
from ..io.utils import file_path_ext, io_element

from .simulation import Simulation

import datetime
ObjectName: TypeAlias = str
DirPath: TypeAlias = str
FileName: TypeAlias = str


@overload
def postprocess_grids(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    load_grid_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    load_numeric_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    cartesian: bool | None = None,
    delete_h5_xdmf: bool = False,
    mode: str = 'a',
    numpy_file_name: str | tuple[str, str] | None = None,
) -> None:
    ...


@overload
def postprocess_grids(
    sim: Simulation,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
    cartesian: bool | None = None,
    delete_h5_xdmf: bool = False,
    mode: str = 'a',
    numpy_file_name: str | tuple[str, str] | None = None,
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
    cartesian: bool | None = None,
    delete_h5_xdmf: bool = False,
    mode: str = 'a',
    numpy_file_name: str | tuple[str, str] | None = None,
):
    grid_series = [i.series for i in sim.solvers if isinstance(i.series, FunctionSeries)]
    numeric_series = [i.series for i in sim.solvers if isinstance(i.series, ConstantSeries)]
    
    if include:
        _include = lambda n: n in include and not n in exclude
    else:
        _include = lambda n: n not in exclude

    dir_path = sim.dir_path
    mesh = sim.t.mesh
    load_grid_args = [
        (i.name, sim.write_file[i.name], io_element(i)) 
        for i in grid_series if _include(i.name)
    ]
    load_numeric_args = [
        (i.name, sim.write_file[i.name], i.shape) 
        for i in numeric_series if _include(i.name)
    ]

    return _postprocess_grids(
        dir_path, 
        mesh, 
        load_grid_args, 
        load_numeric_args, 
        cartesian,
        delete_h5_xdmf, 
        mode,
        numpy_file_name,
    )


@_postprocess_grids.register(str)
def _(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    load_grid_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    load_numeric_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    cartesian: bool | None = None,
    delete_h5_xdmf: bool = False,
    mode: str = 'a',
    numpy_file_name: str | tuple[str, str] | None = None,
):
    if not isinstance(mesh, Mesh):
        mesh_name, mesh_file_name = mesh
        mesh = load_mesh(mesh_name, dir_path, mesh_file_name)
    else:
        mesh = mesh

    cell_type = mesh.topology.cell_name()

    if cartesian is None:
        cartesian = is_cartesian(mesh)

    match cell_type, cartesian:
        case CellType.TRIANGLE, False:
            NpSeries = TriangulationSeries
        case CellType.TRIANGLE | CellType.QUADRILATERAL, True:
            NpSeries = GridSeries
        case CellType.QUADRILATERAL, False:
            raise UnstructuredQuadError
        case _:
            raise ValueError
    
    if numpy_file_name is None:
        numpy_file_name = (
            NpSeries.__name__,
            NumericSeries.__name__,
        )
    if isinstance(numpy_file_name, str):
        numpy_file_name = (numpy_file_name, numpy_file_name)
    grid_file_name, numeric_file_name = numpy_file_name

    for i in load_grid_args:
        name, file_name = i[:2]
        elem = i[2:]
        print(f'loading {name} ...', datetime.datetime.now(), flush=True)
        u = load_function_series(name, dir_path, file_name, mesh, *elem)
        print(f'loaded {name} successfully', datetime.datetime.now(), flush=True)
        # _u2 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u3 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u4 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u5 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u6 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u7 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u8 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u9 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u10 = load_function_series(name, dir_path, file_name, mesh, *elem)
        # _u11 = load_function_series(name, dir_path, file_name, mesh, *elem)
        write(
            NpSeries.from_series(u), 
            file_name=grid_file_name, 
            dir_path=dir_path,
            mode=mode,
        )
        print(f'loaded {name} written', datetime.datetime.now(), flush=True)
        
    for i in load_numeric_args:
        name, file_name = i[:2]
        shape = i[2:]
        print(f'loading {name} ...', datetime.datetime.now(), flush=True)
        c = load_constant_series(name, dir_path, file_name, mesh, *shape)
        print(f'loaded {name} successfully', datetime.datetime.now(), flush=True)
        write(
            NumericSeries.from_series(c), 
            file_name=numeric_file_name, 
            dir_path=dir_path,
            mode=mode,
        )
        print(f' {name} written', datetime.datetime.now(), flush=True)

    if delete_h5_xdmf:
        file_names = set((fn for _, fn, *_ in (*load_grid_args, *load_numeric_args)))
        file_paths = [
            *(file_path_ext(dir_path, fn, 'h5', mkdir=False) for fn in file_names),
            *(file_path_ext(dir_path, fn, 'xdmf', mkdir=False) for fn in file_names),
        ]          
        for fp in file_paths:
            os.remove(fp)



