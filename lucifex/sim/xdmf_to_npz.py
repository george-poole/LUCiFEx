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




ObjectName: TypeAlias = str
DirPath: TypeAlias = str
FileName: TypeAlias = str

@overload
def xdmf_to_npz(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    load_function_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    load_numeric_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    cartesian: bool | None = None,
    delete_xdmf: bool = False,
    mode: str = 'a',
    file_name: str | tuple[str, str] | None = None,
) -> None:
    ...


@overload
def xdmf_to_npz(
    sim: Simulation,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
    cartesian: bool | None = None,
    delete_xdmf: bool = False,
    mode: str = 'a',
    file_name: str | tuple[str, str] | None = None,
) -> None:
    ...


def xdmf_to_npz(*args, **kwargs) -> None:
    """
    Postprocess a simulation dataset exisiting as `.h5` and `.xdmf` 
    files to `.npz` files in preparation for further I/O or 
    postprocessing with `numpy`.
    """
    return _xdmf_to_npz(*args, **kwargs)
    

@singledispatch
def _xdmf_to_npz(arg, *_, **__):
    raise MultipleDispatchTypeError(arg, _xdmf_to_npz)


@_xdmf_to_npz.register(Simulation)
def _(
    sim: Simulation,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
    cartesian: bool | None = None,
    delete_xdmf: bool = False,
    mode: str = 'a',
    file_name: str | tuple[str, str] | None = None,
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

    return _xdmf_to_npz(
        dir_path, 
        mesh, 
        load_grid_args, 
        load_numeric_args, 
        cartesian,
        delete_xdmf, 
        mode,
        file_name,
    )


@_xdmf_to_npz.register(str)
def _(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    load_function_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    load_numeric_args: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    cartesian: bool | None = None,
    delete_xdmf: bool = False,
    mode: str = 'a',
    file_name: str | tuple[str, str] | None = None,
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
    
    if file_name is None:
        file_name = (
            NpSeries.__name__,
            NumericSeries.__name__,
        )
    if isinstance(file_name, str):
        file_name = (file_name, file_name)
    grid_file_name, numeric_file_name = file_name

    for i in load_function_args:
        name, file_name = i[:2]
        elem = i[2:]
        u = load_function_series(name, dir_path, file_name, mesh, *elem)
        write(
            NpSeries.from_series(u), 
            file_name=grid_file_name, 
            dir_path=dir_path,
            mode=mode,
        )
        
    for i in load_numeric_args:
        name, file_name = i[:2]
        shape = i[2:]
        c = load_constant_series(name, dir_path, file_name, mesh, *shape)
        write(
            NumericSeries.from_series(c), 
            file_name=numeric_file_name, 
            dir_path=dir_path,
            mode=mode,
        )

    if delete_xdmf:
        file_names = set((fn for _, fn, *_ in (*load_function_args, *load_numeric_args)))
        file_paths = [
            *(file_path_ext(dir_path, fn, 'h5', mkdir=False) for fn in file_names),
            *(file_path_ext(dir_path, fn, 'xdmf', mkdir=False) for fn in file_names),
        ]          
        for fp in file_paths:
            os.remove(fp)



