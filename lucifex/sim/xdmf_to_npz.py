import os
from collections.abc import Iterable
from typing import overload
from functools import singledispatch

from dolfinx.mesh import Mesh

from ..utils.py_utils import MultipleDispatchTypeError
from ..fdm import ConstantSeries, FunctionSeries
from ..fdm.fdm2npy import (
    as_npy_function_series, as_npy_constant_series,
)
from ..io import write
from ..io.utils import file_path_ext, xdmf_element
from ..io._proxy import ObjectName, FileName
from .sim2io import SimulationFromXDMF
from .simulation import Simulation



@overload
def xdmf_to_npz(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    function_series: Iterable[str],
    constant_series: Iterable[str],
    write_file: str | dict[str, str] | None = None,
    config_file: str = 'CONFIG',
    checkpoint_file: str = 'CHECKPOINT',
    timing_file: str = 'TIMING',
    lazy: bool = True,
    load_args: dict[str, tuple] | None = None,
    *,
    delete_xdmf: bool = False,
    npz_write_file = None # TODO 
) -> None:
    ...


@overload
def xdmf_to_npz(
    sim: Simulation,
    lazy: bool = True,
    load_args: dict[str, tuple] | None = None,
    *,
    delete_xdmf: bool = False,
    npz_write_file = None, 
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
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
    lazy: bool = True,
    load_args: dict[str, tuple] | None = None,
    *,
    delete_xdmf: bool = False,
    npz_write_file = None, 
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
):  
    function_series = [i.name for i in sim.solutions if isinstance(i, FunctionSeries)]
    constant_series = [i.name for i in sim.solutions if isinstance(i, ConstantSeries)]

    if load_args is None:
        load_args = {name: (xdmf_element(sim[name]), ) for name in function_series}

    return _xdmf_to_npz(
        sim.dir_path,
        sim.mesh,
        function_series,
        constant_series,
        sim.write_file,
        sim.config_file,
        sim.checkpoint_file,
        sim.timing_file,
        lazy,
        load_args,
        delete_xdmf=delete_xdmf,
        npz_write_file=npz_write_file,
        exclude=exclude,
        include=include,
    )


@_xdmf_to_npz.register(str)
def _(
    dir_path: str,
    mesh: Mesh | tuple[ObjectName, FileName],
    function_series: Iterable[str],
    constant_series: Iterable[str],
    write_file: str | dict[str, str] | None = None,
    config_file: str = 'CONFIG',
    checkpoint_file: str = 'CHECKPOINT',
    timing_file: str = 'TIMING',
    lazy: bool = True,
    load_args: dict[str, tuple] | None = None,
    *,
    delete_xdmf: bool = False,
    npz_write_file = None,        # TODO 
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
):
    if include:
        _include = lambda n: n in include and not n in exclude
    else:
        _include = lambda n: n not in exclude

    function_series = [i for i in function_series if _include(i)]
    constant_series = [i for i in constant_series if _include(i)]
    
    sim_loader = SimulationFromXDMF(
        dir_path,
        mesh,
        function_series,
        constant_series,
        write_file,
        config_file,
        checkpoint_file,
        timing_file,
        lazy,
        load_args,
    )

    for name in function_series:
        u = sim_loader[name]
        u_npy = as_npy_function_series(u)
        write(u_npy, dir_path=dir_path)

    for name in constant_series:
        c = sim_loader[name]
        c_npy = as_npy_constant_series(c)
        write(c_npy, dir_path=dir_path)

    if delete_xdmf:
        file_names = set(sim_loader.write_file.values())
        file_paths = [
            *(file_path_ext(dir_path, fn, 'h5', mkdir=False) for fn in file_names),
            *(file_path_ext(dir_path, fn, 'xdmf', mkdir=False) for fn in file_names),
        ]          
        for fp in file_paths:
            os.remove(fp)


    # function_series = [i.solution_series for i in sim.solvers if isinstance(i.solution_series, FunctionSeries)]
    # function_series.extend([i.correction_series for i in sim.solvers if i.correction_series is not None])
    # constant_series = [i.solution_series for i in sim.solvers if isinstance(i.solution_series, ConstantSeries)]
    
    # if include:
    #     _include = lambda n: n in include and not n in exclude
    # else:
    #     _include = lambda n: n not in exclude

    # dir_path = sim.dir_path
    # mesh = sim.t.mesh
    # load_grid_args = [
    #     (i.name, sim.write_file[i.name], io_element(i)) 
    #     for i in function_series if _include(i.name)
    # ]
    # constant_series_metadata = [
    #     (i.name, sim.write_file[i.name], i.ufl_shape) 
    #     for i in constant_series if _include(i.name)
    # ]

    # return _xdmf_to_npz(
    #     dir_path, 
    #     mesh, 
    #     load_grid_args, 
    #     constant_series_metadata, 
    #     cartesian,
    #     delete_xdmf, 
    #     mode,
    #     npz_name,
    # )

    # function_series_metadata: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    # constant_series_metadata: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    # grid: bool | None = None,
    # delete_xdmf: bool = False,
    # mode: str = 'a',
    # npz_name: str | tuple[str, str] | None = None,
    # load_args: dict[str, tuple] | None = None


    # if not isinstance(mesh, Mesh):
    #     mesh_name, mesh_file_name = mesh
    #     mesh = load_mesh(mesh_name, dir_path, mesh_file_name)
    # else:
    #     mesh = mesh

    # simplicial = is_simplicial(mesh)

    # if grid is None:
    #     grid = is_grid(mesh)

    # match simplicial, grid:
    #     case True, False:
    #         NpSeries = TriFunctionSeries
    #     case _, True:
    #         NpSeries = GridFunctionSeries
    #     case False, False:
    #         raise QuadNonGridMeshError('Conversion to `numpy` data')
    #     case _:
    #         raise ValueError
    
    # if npz_name is None:
    #     npz_name = (
    #         NpSeries.__name__,
    #         NumericSeries.__name__,
    #     )
    # if isinstance(npz_name, str):
    #     npz_name = (npz_name, npz_name)
    # funcs_npz_name, consts_npz_name = npz_name

    # for i in function_series_metadata:
    #     name, file_name = i[:2]
    #     elem = i[2:]
    #     u = load_function_series(name, dir_path, file_name, mesh, *elem)
    #     write(
    #         NpSeries.from_series(u), 
    #         file_name=funcs_npz_name, 
    #         dir_path=dir_path,
    #         mode=mode,
    #     )
        
    # for i in constant_series_metadata:
    #     name, file_name = i[:2]
    #     shape = i[2:]
    #     c = load_constant_series(name, dir_path, file_name, mesh, *shape)
    #     write(
    #         NumericSeries.from_series(c), 
    #         file_name=consts_npz_name, 
    #         dir_path=dir_path,
    #         mode=mode,
    #     )


