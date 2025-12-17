import os
from collections.abc import Iterable
from typing import overload
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
from ..io.proxy import ObjectName, FileName
from .simulation import Simulation


@overload
def xdmf_to_npz(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    function_series_metadata: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    constant_series_metadata: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    cartesian: bool | None = None,
    delete_xdmf: bool = False,
    mode: str = 'a',
    npz_name: str | tuple[str, str] | None = None,
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
    npz_name: str | tuple[str, str] | None = None,
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
    npz_name: str | tuple[str, str] | None = None,
):
    function_series = [i.series for i in sim.solvers if isinstance(i.series, FunctionSeries)]
    function_series.extend([i.correction_series for i in sim.solvers if i.correction_series is not None])
    constant_series = [i.series for i in sim.solvers if isinstance(i.series, ConstantSeries)]
    
    if include:
        _include = lambda n: n in include and not n in exclude
    else:
        _include = lambda n: n not in exclude

    dir_path = sim.dir_path
    mesh = sim.t.mesh
    load_grid_args = [
        (i.name, sim.write_file[i.name], io_element(i)) 
        for i in function_series if _include(i.name)
    ]
    constant_series_metadata = [
        (i.name, sim.write_file[i.name], i.shape) 
        for i in constant_series if _include(i.name)
    ]

    return _xdmf_to_npz(
        dir_path, 
        mesh, 
        load_grid_args, 
        constant_series_metadata, 
        cartesian,
        delete_xdmf, 
        mode,
        npz_name,
    )


@_xdmf_to_npz.register(str)
def _(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    function_series_metadata: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    constant_series_metadata: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    cartesian: bool | None = None,
    delete_xdmf: bool = False,
    mode: str = 'a',
    npz_name: str | tuple[str, str] | None = None,
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
    
    if npz_name is None:
        npz_name = (
            NpSeries.__name__,
            NumericSeries.__name__,
        )
    if isinstance(npz_name, str):
        npz_name = (npz_name, npz_name)
    funcs_npz_name, consts_npz_name = npz_name

    for i in function_series_metadata:
        name, file_name = i[:2]
        elem = i[2:]
        u = load_function_series(name, dir_path, file_name, mesh, *elem)
        write(
            NpSeries.from_series(u), 
            file_name=funcs_npz_name, 
            dir_path=dir_path,
            mode=mode,
        )
        
    for i in constant_series_metadata:
        name, file_name = i[:2]
        shape = i[2:]
        c = load_constant_series(name, dir_path, file_name, mesh, *shape)
        write(
            NumericSeries.from_series(c), 
            file_name=consts_npz_name, 
            dir_path=dir_path,
            mode=mode,
        )

    if delete_xdmf:
        file_names = set((fn for _, fn, *_ in (*function_series_metadata, *constant_series_metadata)))
        file_paths = [
            *(file_path_ext(dir_path, fn, 'h5', mkdir=False) for fn in file_names),
            *(file_path_ext(dir_path, fn, 'xdmf', mkdir=False) for fn in file_names),
        ]          
        for fp in file_paths:
            os.remove(fp)



