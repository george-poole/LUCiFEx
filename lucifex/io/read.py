from functools import singledispatch
from typing import overload, Iterable, Protocol, Any

import h5py
import numpy as np

from ..fem import Constant, Function, is_unsolved
from ..fdm import FunctionSeries, ConstantSeries
from ..utils import (
    MultipleDispatchTypeError, 
    is_continuous_lagrange, 
    is_discontinuous_lagrange, 
    as_slice,
    StrSlice,
)
from .utils import file_path_ext, dofs_array_dim


class ReadObject(Protocol):
    def __call__(
        self, 
        obj: Any,
        dir_path: str | None = None,
        file_name: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        ...



@overload
def read(
    u: Function | Constant,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> None:
    """
    `Function` objects must have a `P₁` or `DP₀` function space
    to be read from file.
    """
    ...


@overload
def read(
    u: FunctionSeries | ConstantSeries,
    dir_path: str | None = None,
    file_name: str | None = None,
    slc: Iterable[int] | StrSlice = slice(0, None),
) -> None:
    """
    `FunctionSeries` objects must be have a `P₁` or `DP₀` function space
    to be read from file.
    """
    ...


def read(
    u: Function | Constant | FunctionSeries | ConstantSeries,
    dir_path = None,
    file_name = None,
    *args,
    **kwargs,
) -> None:
    
    if file_name is None:
        file_name = u.name

    file_path = file_path_ext(dir_path, file_name, 'h5', mkdir=False)

    return _read(u, file_path, *args, **kwargs)


@singledispatch
def _read(u, *_, **__):
    raise MultipleDispatchTypeError(u, _read)


@_read.register(np.ndarray)
def _(
    u: np.ndarray,
    *,
    dofs: np.ndarray | None = None,
):

    dim = len(dofs)
    dim_expected = dofs_array_dim(u.shape)
    if not dim == dim_expected:
        ValueError(f'Cannot load data of length {dim} into object of shape {u.shape}.')

    match u.shape:
        case ():
            u[()] = dofs[0]
        case (_, ):
            u[:] = dofs[:]
        case (_, _):
            u[:] = np.reshape(dofs, u.shape)
        case _:
            raise NotImplementedError


@_read.register(Function)
@_read.register(Constant)
def _(
    u: Function | Constant,
    file_path,
    *,
    dofs=None,
):
    if dofs is None:
        dim = dofs_array_dim(u.ufl_shape)
        with h5py.File(file_path, "r") as h5:
            dofs = _dofs_time_series(h5, u.name, dim)[0][0]

    if isinstance(u, Function):
        return _read(u.x.array, dofs=dofs)
    else:
        return _read(u.value, dofs=dofs)


@_read.register(ConstantSeries)
@_read.register(FunctionSeries)
def _(
    u: ConstantSeries | FunctionSeries,
    file_path,
    slc: Iterable[int] | StrSlice | int = slice(0, None),
):      
    if isinstance(slc, int):
        slc = range(0, len(u.series), slc)
    else:
        slc = as_slice(slc)

    if isinstance(u, ConstantSeries):
        container = Constant(u.mesh, name=u.name, shape=u.shape)
    elif isinstance(u, FunctionSeries):
        assert is_continuous_lagrange(u.function_space, 1) or is_discontinuous_lagrange(u.function_space, 0)
        container = Function(u.function_space, name=u.name)
    else:
        raise MultipleDispatchTypeError(u)
    dim = dofs_array_dim(container.ufl_shape)

    if is_unsolved(u[u.FUTURE_INDEX - 1]):
        future = False
    else:
        future = True

    _store = u.store
    u.store = 1

    with h5py.File(file_path, "r") as h5:
        if isinstance(slc, slice):
            n_start = slc.start
            n_stop = slc.stop
            n_step = slc.step
            if n_start is None:
                n_start = 0
            if n_stop is None:
                n_stop = len(h5[f"Function/{u.name}"])
            if n_step is None:
                n_step = 1
            time_indices = range(n_start, n_stop, n_step)
        else:
            time_indices = slc
        dofs_series, time_series = _dofs_time_series(h5, u.name, dim)
        for n in time_indices:
            _read(container, file_path, dofs=dofs_series[n])
            u.update(container, future)
            u.forward(time_series[n])

    u.store = _store


def _dofs_time_series(
    h5: h5py.File, 
    u_name: str,
    dim: int | None
) -> tuple[list[np.ndarray], list[float]]:
    """
    `([𝐱₀, 𝐱₁, 𝐱₂, ...], [t₀, t₁, t₂, ...])`
    """
    tstr_dofs: dict[str, np.ndarray] = h5[f"Function/{u_name}"]
    t_dofs: dict[float, np.ndarray] = dict(sorted({
        float(k.replace("_", ".")): v for k, v in tstr_dofs.items()
    }.items()))
    t = [i for i in t_dofs.keys()]
    if dim is None:
        dofs = [x[:, 0] for x in t_dofs.values()]
    else:
        dofs = [np.array([j for i in x[:, :dim] for j in i]) for x in t_dofs.values()]
    return dofs, t

