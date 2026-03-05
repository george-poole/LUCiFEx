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
from .sim2io import SimulationFromXDMF
from .simulation import Simulation


@overload
def xdmf_to_npz(
    dir_path: str,
    mesh: Mesh | tuple[str, str],
    function_series: Iterable[str],
    constant_series: Iterable[str],
    write_file: str | dict[str, str] | None = None,
    parameter_file: str = 'PARAMETERS',
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
    use_cache: bool = True,
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
        load_args = {
            **{name: (xdmf_element(sim[name]), ) for name in function_series},
            **{name: (sim[name].ufl_shape, ) for name in constant_series},
        }

    return _xdmf_to_npz(
        sim.dir_path,
        sim.mesh,
        function_series,
        constant_series,
        sim.write_file,
        sim.parameter_file,
        sim.checkpoint_file,
        sim.timing_file,
        lazy,
        use_cache,
        load_args,
        delete_xdmf=delete_xdmf,
        npz_write_file=npz_write_file,
        exclude=exclude,
        include=include,
    )


@_xdmf_to_npz.register(str)
def _(
    dir_path: str,
    mesh: Mesh | tuple[str, str],
    function_series: Iterable[str],
    constant_series: Iterable[str],
    write_file: str | dict[str, str] | None = None,
    parameter_file: str = 'PARAMETERS',
    checkpoint_file: str = 'CHECKPOINT',
    timing_file: str = 'TIMING',
    lazy: bool = True,
    use_cache: bool = True,
    load_args: dict[str, tuple] | None = None,
    *,
    delete_xdmf: bool = False,
    npz_write_file: dict[str, str] | str | None = None,
    exclude: Iterable[str] = (),
    include: Iterable[str] = (),
):
    if include:
        _include = lambda n: n in include and not n in exclude
    else:
        _include = lambda n: n not in exclude

    function_series = [i for i in function_series if _include(i)]
    constant_series = [i for i in constant_series if _include(i)]
    
    sim_from_xdmf = SimulationFromXDMF(
        dir_path,
        mesh,
        function_series,
        constant_series,
        write_file,
        parameter_file,
        checkpoint_file,
        timing_file,
        lazy,
        use_cache,
        load_args,
    )

    if npz_write_file is None:
        npz_write_file = sim_from_xdmf.write_file
    if isinstance(npz_write_file, str):
        npz_write_file = {
            **dict(zip(function_series, [write_file] * len(function_series))),
            **dict(zip(constant_series, [write_file] * len(constant_series)))
        }

    for name in function_series:
        u = sim_from_xdmf[name]
        u_npy = as_npy_function_series(u)
        write(u_npy, npz_write_file[name], dir_path)

    for name in constant_series:
        c = sim_from_xdmf[name]
        c_npy = as_npy_constant_series(c)
        write(c_npy, npz_write_file[name], dir_path)

    if delete_xdmf:
        file_names = set(sim_from_xdmf.write_file.values())
        file_paths = [
            *(file_path_ext(dir_path, fn, 'h5', mkdir=False) for fn in file_names),
            *(file_path_ext(dir_path, fn, 'xdmf', mkdir=False) for fn in file_names),
        ]          
        for fp in file_paths:
            os.remove(fp)