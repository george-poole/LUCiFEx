from collections.abc import Callable
from inspect import Signature, signature
from typing import Concatenate, ParamSpec, overload, TypeVar, TypeAlias

from dolfinx.fem import Constant, Function

from ..fdm import FunctionSeries, ConstantSeries
from ..io import write
from ..utils.deferred import Stopper, Writer, defer
from .simulation import Simulation


T = TypeVar('T', FunctionSeries, ConstantSeries, Function, Constant)
P = ParamSpec('P')
CreateStopper: TypeAlias = Callable[[Simulation], Stopper]

@overload
def stopper(
    u: T,
    condition: Callable[Concatenate[T, P], bool],
) -> Callable[P, Stopper]:
    ...

@overload
def stopper(
    u: str,
    condition: Callable[Concatenate[T, P], bool],
) -> Callable[P, CreateStopper]:
    ...

def stopper(
    u: str | T,
    condition: Callable[Concatenate[T, P], bool],
):  
    paramspec_params = list(signature(condition).parameters.values())[1:]

    def _(*args: P.args, **kwargs: P.kwargs):
        if isinstance(u, str):
            _.__signature__ = Signature(paramspec_params, return_annotation=Callable[[Simulation], Stopper])
            return lambda sim: stopper(sim[u], condition)(*args, **kwargs)
        else:
            _.__signature__ = Signature(paramspec_params, return_annotation=Stopper)
            return Stopper(defer(condition)(u, *args, **kwargs))

    return _
            

def as_stopper(
    arg: Stopper | Callable[[], bool] | CreateStopper,
    sim: Simulation,
) -> Stopper:
    if isinstance(arg, Stopper):
        return arg
    elif has_simulation_arg(arg):
        return arg(sim)
    else:
        return Stopper(arg)


T = TypeVar('T', FunctionSeries, ConstantSeries, Function, Constant)
P = ParamSpec('P')
CreateWriter: TypeAlias = Callable[[Simulation], Writer]

@overload
def writer(
    u: T,
    condition: Callable[Concatenate[T, P], bool],
    routine: Callable[[float, T], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> Callable[P, Writer]:
    ...


@overload
def writer(
    u: T,
    condition: int | float | None = None,
    routine: Callable[[float, T], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> Writer:
    ...


@overload
def writer(
    u: str,
    condition: Callable[Concatenate[T, P], bool],
    routine: Callable[[float, T], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> Callable[P, CreateWriter]:
    ...

@overload
def writer(
    u: str,
    condition: int | float | None = None,
    routine: Callable[[float, T], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> CreateWriter:
    ...


def writer(
    u: str | T,
    condition: Callable[Concatenate[T, P], bool] | int | float |  None = None,
    routine: Callable[[float, T], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
):
    if isinstance(condition, (int, float)) or condition is None:
        if isinstance(u, str):
            def _(sim: Simulation) -> Stopper:
                return writer(
                sim[u], 
                condition, 
                routine, 
                sim.dir_path if dir_path is None else dir_path,
                file_name,
            )
            return _
        else:
            if routine is None:
                _routine = lambda t: write(u, file_name, dir_path, t, 'a', u.mesh.comm)
            else:
                _routine = lambda t: routine(t, u)
            return Writer(_routine, condition, u)
    else:
        paramspec_params = list(signature(condition).parameters.values())[1:]
        def _(
            *args: P.args,
            **kwargs: P.kwargs,
        ):
            if isinstance(u, str):
                _.__signature__ = Signature(paramspec_params, return_annotation=Callable[[Simulation], Writer])
                return lambda sim: writer(
                    sim[u], 
                    condition, 
                    routine, 
                    sim.dir_path if dir_path is None else dir_path,
                    file_name,
                )(*args, **kwargs)
            else:
                _.__signature__ = Signature(paramspec_params, return_annotation=Writer)
                if routine is None:
                    _routine = lambda t: write(u, file_name, dir_path, t, 'a', u.mesh.comm)
                else:
                    _routine = lambda t: routine(t, u)
                return Writer(_routine, defer(condition)(u, *args, **kwargs), u.name)
        return _


def as_writer(
    arg: Writer | Callable[[], bool] | CreateWriter,
    sim: Simulation,
) -> Writer:
    if isinstance(arg, Writer):
        return arg
    elif has_simulation_arg(arg):
        return arg(sim)
    else:
        return Writer(arg)
    

def has_simulation_arg(func: Callable) -> bool:
    assert callable(func)
    if len(signature(func).parameters) != 1:
        return False
    return list(signature(func).parameters.values())[0].annotation is Simulation