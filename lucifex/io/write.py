from functools import singledispatch, lru_cache
import os
import pickle as pkl
from typing import Literal, overload, Protocol, Any, Callable, Iterable

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from ufl.core.expr import Expr
from dolfinx.io import XDMFFile
from dolfinx.mesh import Mesh, create_interval
from dolfinx.fem import Function, Constant, FunctionSpace, VectorFunctionSpace
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..utils import extract_mesh, MultipleDispatchTypeError, set_fem_function, StrSlice, as_slice
from ..fem import LUCiFExConstant
from ..fdm.series import FunctionSeries, ConstantSeries, GridSeries, NumericSeries

from .utils import file_path_ext, io_array_dim


class WriteFunctionType(Protocol):
    def __call__(
        self, 
        obj: Any,
        file_name: str,
        dir_path: str,
        # t: Any | None = None,
        *args,
        **kwargs,
    ) -> None:
        ...


@overload
def write(
    u: Function | LUCiFExConstant | Mesh,
    file_name: str | None = None,
    dir_path: str | None = None,
    t: float | Constant | None = None,
    mode: Literal["a", "w", "c"] = "a",
    comm: MPI.Comm | None = None,
) -> None:
    ... # TODO what conditions for xdmf interpolation to P1 vs DP0?


@overload
def write(
    u: Expr,
    file_name: str | None = None,
    dir_path: str | None = None,
    t: float | Constant | None = None,
    mode: Literal["a", "w", "c"] = "a",
    comm: MPI.Comm | None = None,
    *,
    mesh: Mesh | None = None,
    fs: tuple[str, int] = ("P", 1),
    name: str = 'expression',
) -> None:
    ...


@overload
def write(
    u: FunctionSeries | ConstantSeries,
    file_name: str | None = None,
    dir_path: str | None = None,
    slc: StrSlice = slice(0, None),
    mode: Literal["a", "w"] = "a",
    comm: MPI.Comm | None = None,
) -> None:
    ...


@overload
def write(
    u: GridSeries,
    file_name: str | None = None,
    dir_path: str | None = None,
    slc: StrSlice = slice(0, None),
    mode: Literal["a", "w"] = "a",
    sep: str = '__',
    axes: tuple[str, ...] = ('x', 'y', 'z'),
) -> None:
    ...


@overload
def write(
    u: NumericSeries,
    file_name: str | None = None,
    dir_path: str | None = None,
    slc: StrSlice = slice(0, None),
    mode: Literal["a", "w"] = "a",
    sep: str = '__',
) -> None:
    ...


@overload
def write(
    namespace: dict[str, Any],
    file_name: str,
    dir_path: str,
    mode: str= 'a',
    sep: str = ' = ',
) -> None:
    ...


@overload
def write(
    namespace: dict[str, float | np.ndarray],
    file_name: str,
    dir_path: str,
    mode: str= 'a',
    **np_savez_kwargs,
) -> None:
    ...


@overload
def write(
    fig: Figure | tuple[Figure, Axes],
    file_name: str | Callable[[Figure], str] | None = None,
    dir_path: str | None = None,
    close: bool = True,
    pickle: bool = True,
    bbox_inches: str | None = "tight",
    dpi: int = 150,
    file_ext: str | Iterable[str] = "pdf",
    **savefig_kwargs,
):
    ...


@overload
def write(
    anim: FuncAnimation,
    file_name: str,
    dir_path: str | None = None,
    file_ext: str = "mp4",
    writer: str = "ffmpeg",
    fps: int | None = None,
    dpi: int = 150,
    bitrate: int | None = None,
    **saveanim_kwargs,
) -> None:
    ...


def write(
    obj,
    file_name=None,
    dir_path=None,
    *args,
    **kwargs,
):
    """
    `Function` objects are written as their
    interpolation to a `P₁` or `DP₀` function space.

    `Constant` objects are written by typecasting
    to a `DP₀` function on an interval mesh.

    `mode='a'` to append \\
    `mode='w'` to write \\
    `mode='c'` to checkpoint
    """
    if file_name is None:
        try:
            file_name = getattr(obj, 'name')
        except AttributeError:
            pass

    if isinstance(file_name, (tuple, list)):
        [_write(i, fn, dir_path, *args, **kwargs) for i, fn in zip(obj, file_name, strict=True)]
    else:
        return _write(obj, file_name, dir_path, *args, **kwargs)


@singledispatch
def _write(u, *_, **__) -> None:
    raise MultipleDispatchTypeError(u, _write)


@_write.register(Mesh)
def _(
    msh: Mesh,
    file_name: str,
    dir_path,
    t=None,
    mode='a',
    comm=None,
):
    file_path = file_path_ext(dir_path, file_name, 'xdmf')

    if t is not None:
        name = msh.name
        msh.name = f"{name}_{float(t)}"

    if comm is None:
        comm = msh.comm

    with XDMFFile(comm, file_path, mode) as xdmf:
        xdmf.write_mesh(msh)

    if t is not None:
        msh.name = name


@_write.register(Function)
def _(
    u: Function,
    file_name: str,
    dir_path = None,
    t = None,
    mode = 'a',
    comm=None,
    *,
    xdmf: XDMFFile | None = None,
):  
    file_path = file_path_ext(dir_path, file_name, 'xdmf')

    if t is None:
        t = 0.0
    t = float(t)

    if comm is None:
        comm = u.function_space.mesh.comm

    if mode == "c":
        dp0_function = _cached_dp0_function(u)
        dp0_function.x.array[:] = u.x.array[:]
        return _write(dp0_function, file_name, dir_path, t, "a", comm)
    
    if mode == 'w':
        _write(u.function_space.mesh, file_name, dir_path, mode=mode)
    else:
        _cached_write_mesh(u.function_space.mesh, file_name, dir_path)
    
    if xdmf is None:
        with XDMFFile(comm, file_path, 'a') as xdmf:
            xdmf.write_function(u, t)
    else:
        xdmf.write_function(u, t)


@_write.register(LUCiFExConstant)
def _(
    c: LUCiFExConstant,
    file_name: str,
    dir_path,
    t=None,
    mode='a',
    comm=None,
    *,
    xdmf: XDMFFile = None,
):
    if comm is None:
        comm = c.mesh.comm
    dp0_function = _cached_dp0_function(c)
    dp0_function.x.array[:] = c.value.flatten()
    return _write(dp0_function, file_name, dir_path, t, mode, comm, xdmf=xdmf)


@_write.register(FunctionSeries)
@_write.register(ConstantSeries)
def _(
    u: FunctionSeries | ConstantSeries,
    file_name: str,
    dir_path,
    slc=slice(0, None),
    mode='a',
    comm=None,
):    
    file_path = file_path_ext(dir_path, file_name, 'xdmf')
    slc = as_slice(slc)

    if comm is None:
        comm = u.mesh.comm

    u_series = u.series[slc]
    t_series = u.time_series[slc]

    if None in t_series:
        t_series = range(len(u.series))

    with XDMFFile(comm, file_path, mode) as xdmf:
        for _u, _t in zip(u_series, t_series, strict=True):
            _write(_u, file_name, dir_path, _t, mode, comm, xdmf=xdmf)
            mode = "a"


@_write.register(GridSeries)
def _(
    u: GridSeries,
    file_name: str,
    dir_path,
    slc=slice(0, None), 
    mode='a',
    sep = '__',
    axes: tuple[str, ...] = ('x', 'y', 'z'),
): 
    file_path = file_path_ext(dir_path, file_name, 'npz')

    slc = as_slice(slc)
    u = GridSeries(u.series[slc], u.time_series[slc], u.axes, u.name)

    d = {}
    if mode == 'a' and os.path.exists(file_path):
        d.update(np.load(file_path).items())
    d.update(dict(zip([_create_npz_key(u.name, t, sep) for t in u.time_series], u.series)))
    d.update({_create_npz_key(u.name, k, sep): v for k, v in zip(axes, u.axes)})
    np.savez(file_path, **d)


@_write.register(NumericSeries)
def _(
    u: NumericSeries,
    file_name: str,
    dir_path,
    slc = slice(0, None), 
    mode = 'a',
    sep = '__',
): 
    file_path = file_path_ext(dir_path, file_name, 'npz')

    slc = as_slice(slc)
    u = NumericSeries(u.series[slc], u.time_series[slc], u.name)

    d = {}
    if mode == 'a' and os.path.exists(file_path):
        d.update(np.load(file_path).items())
    d.update(dict(zip([_create_npz_key(u.name, t, sep) for t in u.time_series], u.series)))
    np.savez(file_path, **d)


def _create_npz_key(name: str, t: float | str, sep: str) -> str:
    return f'{name}{sep}{str(t)}'


@_write.register(Expr)
def _(
    u: Expr,
    file_name,
    dir_path,
    t=None,
    mode='a',
    comm=None,
    *,
    mesh=None,
    fam_deg=("P", 1),
    name='expression',
):
    if mesh is None:
        mesh = extract_mesh(u)

    if comm is None:
        comm = mesh.comm

    shape = u.ufl_shape
    if shape == ():
        elem = fam_deg
    else:
        if len(shape) > 1:
            raise NotImplementedError
        dim = shape[0]
        elem = (*fam_deg, dim)
    fs = _cached_function_space(mesh, elem)
    func = Function(fs, name=name)
    func.name = name
    set_fem_function(func, u)
    return write(func, file_name, dir_path, t, mode, comm)


@_write.register(dict)
def _(
    namespace: dict[str, Any] | dict[str, float | np.ndarray],
    file_name: str,
    dir_path,
    mode: str= 'a',
    file_ext: str | None = None,
    **kwargs,
) -> None:
    
    NPZ = 'npz'
    TXT = 'txt'
    
    if file_ext is None:
        if all(isinstance(i, (float, np.ndarray)) for i in namespace.values()):
            file_ext = NPZ
        else:
            file_ext = TXT

    file_path = file_path_ext(dir_path, file_name, file_ext)

    if file_ext == TXT:
        _kwargs = {'sep': ' = '}
        _kwargs.update(kwargs)
        with open(file_path, mode) as f:
            for k, v in namespace.items():
                print(k, repr(v), **_kwargs, file=f)
    elif file_ext == NPZ:
        array_dict = {}
        if mode == 'a' and os.path.isfile(file_path):
            array_dict.update(np.load(file_path).items())
        array_dict.update(namespace)
        np.savez(file_path, **array_dict, **kwargs)
    else:
        raise ValueError
    

@_write.register(float)
@_write.register(np.ndarray)
def _(
    u,
    file_name: str,
    dir_path,
    name,
    mode= 'a',
):
    if isinstance(name, str):
        d = {name: u}
    else:
        d = {n: v for n, v in zip(name, u, strict=True)} 
    return _write(d, file_name, dir_path, mode)


@_write.register(Figure)
def _(
    fig: Figure,
    file_name,
    dir_path,
    close = True,
    pickle = True,
    bbox_inches = "tight",
    dpi = 150,
    file_ext = "pdf",
    **kwargs,
):
    if not isinstance(file_ext, str):
        assert all(isinstance(i, str) for i in file_ext)
        for ext in file_ext:
            _write(fig, file_name, dir_path, close, pickle, bbox_inches, dpi, ext, **kwargs)
        return
        
    if callable(file_name):
        file_name = file_name(fig)
    assert file_name is not None
    file_path = file_path_ext(dir_path, file_name, file_ext)
    fig.savefig(file_path, bbox_inches=bbox_inches, dpi=dpi, **kwargs)
    if pickle:
        pkl_path = f'{os.path.splitext(file_path)[0]}.pickle'
        with open(pkl_path, 'wb') as f:
            pkl.dump(fig, f) 
    if close:
        plt.close(fig)


@_write.register(tuple)
@_write.register(list)
def _(
    obj,
    *args,
    **kwargs,
):
    
    is_fig_ax = lambda obj: len(obj) == 2 and isinstance(obj[0], Figure) and isinstance(obj[1], Axes) 
    if all(isinstance(i, float) for i in obj):
        return _write(np.array(obj), *args, **kwargs)
    elif is_fig_ax(obj):
        return _write(obj[0], *args, **kwargs)
    elif all(isinstance(i, tuple) and is_fig_ax(i) for i in obj):
        [_write(i, *args, **kwargs) for i in obj]
        return
    else:
        for t in _write.registry.keys():
            if t is not object and all(isinstance(i, t) for i in obj):
                [_write(i, *args, **kwargs) for i in obj]
                return
    raise TypeError
    

@_write.register(FuncAnimation)
def _(
    anim: FuncAnimation,
    file_name: str,
    dir_path=None,
    file_ext="mp4",
    writer= "ffmpeg",
    fps= None,
    dpi=150,
    bitrate = None,
    **saveanim_kwargs,
) -> None:
    file_path = file_path_ext(dir_path, file_name, file_ext)
    anim.save(file_path, writer, fps, dpi, bitrate=bitrate, **saveanim_kwargs)


@lru_cache
def _cached_dp0_function(
    c: Function | LUCiFExConstant,
) -> Function:
    if isinstance(c, LUCiFExConstant):
        mesh = c.mesh
        dim = io_array_dim(c.ufl_shape)
    elif isinstance(c, Function):
        mesh = c.function_space.mesh
        dim = len(c.x.array)
    else:
        raise TypeError

    mesh = _cached_interval(mesh, dim, c.name)
    fs = _cached_function_space(mesh, ('DP', 0))
    # mesh = create_interval(comm, dim, (0, dim))
    # mesh.name = c.name
    # fs = FunctionSpace(mesh, ("DP", 0))
    f = Function(fs)
    f.name = c.name
    return f


@lru_cache
def _cached_interval(
    mesh: Mesh,
    dim: int,
    name: str,
) -> Mesh:
    interval = create_interval(mesh.comm, dim, (0, dim))
    interval.name = name
    return interval


@lru_cache
def _cached_function_space(
    mesh: Mesh,
    elem: tuple,
) -> FunctionSpace:
    if len(elem) == 2:
        fam, deg = elem
        return FunctionSpace(mesh, (fam, deg))
    elif len(elem) == 3:
        fam, deg, dim = elem
        return VectorFunctionSpace(mesh, (fam, deg), dim)
    else:
        raise ValueError
    

@lru_cache(maxsize=None)
def _cached_write_mesh(mesh, file_name, dir_path) -> None:
    """Ensuring that a time-independent mesh is written once only"""
    _write(mesh, file_name, dir_path, mode='w', comm=mesh.comm)


def clear_write_cache():
    _cached_write_mesh.cache_clear()
    _cached_dp0_function.cache_clear()
    _cached_function_space.cache_clear()


# @_write.register(Image.Image)
# def _(
#     img: Image.Image,
#     file_name,
#     dir_path,
#     close = False,
#     file_ext = "jpg",
#     **kwargs,
# ):
#     file_path = file_path_ext(dir_path, file_name, file_ext)
#     img.save(file_path, **kwargs)
#     if close:
#         img.close()



# @overload
# def write(
#     u: Iterable[float],
#     t: float | Constant | None = None,
#     file_name: str | None = None,
#     dir_path: str | None = None,
#     mode: Literal["a", "w", "c"] = "a",
#     comm: MPI.Comm = MPI.COMM_WORLD,
#     *,
#     mesh: Mesh,
#     name: str | None = None,
# ):
#     ...


# @_write.register(np.ndarray)
# @_write.register(tuple)
# @_write.register(list)
# @_write.register(Iterable)
# def _(
#     u: Iterable[float],
#     t,
#     file_path,
#     mode,
#     comm=None,
#     *,
#     mesh,
#     name=None,
# ):
#     c = StaticConstant(mesh, np.array(u))
#     if name is None:
#         name = "constant"
#     c.name = name
#     return write(c, t, file_path, mode, comm)

# def write_to(_write_to):
#     @wraps(_write_to)
#     def wrapper(t, series, condition, *args, **kwargs):
#         if len({i.name for i in series}) != len(series):
#             raise RuntimeError("Series' names must be unique")

#         if isinstance(condition, int):
#             _condition = lambda n: n % condition == 0
#         else:
#             _condition = condition

#         return _write_to(t, series, _condition, *args, **kwargs)

#     return wrapper


# @write_to
# def write_to_collective(
#     t: ConstantSeries,
#     series: list[FunctionSeries | ConstantSeries],
#     condition: int | Callable[[int], bool],
#     file_name: str = None,
#     dir_path: str = None,
#     comm: MPI.Comm = MPI.COMM_WORLD,
#     checkpoint: bool = False,
# ) -> Callable[[int], None]:
#     if file_name is None:
#         file_name = "_".join([i.name for i in series])

#     def _io_routine(n):
#         if condition(n) == 0:
#             mode = "w" if n == 0 and not checkpoint else "a"
#             write_mesh = True if n == 0 and not checkpoint else False
#             for i in series:
#                 write(
#                     i[0], t[0], file_name, dir_path, mode, comm, write_mesh=write_mesh
#                 )
#                 mode = "a"

#     return _io_routine


# @write_to
# def write_to_individual(
#     t: ConstantSeries,
#     series: list[FunctionSeries | ConstantSeries],
#     condition: int | Callable[[int], bool],
#     file_name: Optional[list[str]] = None,
#     dir_path: str = None,
#     comm: MPI.Comm = MPI.COMM_WORLD,
#     checkpoint: bool = False,
# ) -> Callable[[int], None]:
#     def _io_routine(n):
#         if condition(n):
#             mode = "w" if n == 0 and not checkpoint else "a"
#             [write(i, t[0], i.name, dir_path, mode, comm) for i in series]

#     return _io_routine


# def multi_mono_writers(self, t, function_series, function_series_2, comm) -> list[Callable[[int], bool]]:
#     writers = []

#     n_io_mono, n_io_multi = self.n_io

#     if n_io_mono is not None:
#         writers.append(write_to_collective(t, [function_series], lambda n: n % n_io_mono == 0, dir_path=self.io_dir, comm=comm, checkpoint=self.checkpoint))
#     if n_io_multi is not None:
#         writers.append(write_multi(t, [function_series_2], cond = lambda n: n % n_io_multi == 0, self.io_dir, comm, self.checkpoint))

#     return writers
