from collections.abc import Callable
from inspect import Signature, signature, Parameter
from typing import (
    Concatenate, ParamSpec, 
    overload, TypeVar, TypeAlias,
)

from ..fem import Constant, Function
from ..fdm import FunctionSeries, ConstantSeries
from ..io import write
from ..utils.py_utils.deferred import Stopper, Writer, create_lazy_evaluator, LazyEvaluator
from .simulation import Simulation


StopperFactory: TypeAlias = Callable[[Simulation], Stopper]

StopperMetaFactory: TypeAlias = Callable[..., StopperFactory]


U = TypeVar('U', FunctionSeries, ConstantSeries, Function, Constant)
P = ParamSpec('P')
@overload
def stopper(
    u: U,
    condition: Callable[Concatenate[U, P], bool],
) -> Callable[P, Stopper]:
    ...

@overload
def stopper(
    u: str,
    condition: Callable[Concatenate[U, P], bool],
) -> Callable[P, StopperFactory]:
    ...


def stopper(
    u: str | U,
    condition: Callable[Concatenate[U, P], bool],
):  
    paramspec_params = list(signature(condition).parameters.values())[1:]

    def _(*args: P.args, **kwargs: P.kwargs):
        if isinstance(u, str):
            _.__signature__ = Signature(paramspec_params, return_annotation=Callable[[Simulation], Stopper])
            return lambda sim: stopper(sim[u], condition)(*args, **kwargs)
        else:
            _.__signature__ = Signature(paramspec_params, return_annotation=Stopper)
            return Stopper(create_lazy_evaluator(condition)(u, *args, **kwargs))

    return _
            

def as_stopper(
    arg: Stopper | LazyEvaluator[bool] | StopperFactory,
    sim: Simulation,
) -> Stopper:
    if isinstance(arg, Stopper):
        return arg
    elif is_controller_factory(arg):
        _stopper = arg(sim)
        if not isinstance(_stopper, Stopper):
            raise TypeError(f'{_stopper} is not a `Stopper`.')
        return _stopper
    else:
        return Stopper(arg)


WriterFactory: TypeAlias = Callable[[Simulation], Writer]

WriterMetaFactory: TypeAlias = Callable[..., WriterFactory]


U = TypeVar('U', FunctionSeries, ConstantSeries, Function, Constant)
P = ParamSpec('P')
@overload
def writer(
    u: U,
    condition: Callable[Concatenate[U, P], bool],
    routine: Callable[[float, U], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> Callable[P, Writer]:
    ...

@overload
def writer(
    u: U,
    condition: int | float | None = None,
    routine: Callable[[float, U], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> Writer:
    ...

@overload
def writer(
    u: str,
    condition: Callable[Concatenate[U, P], bool],
    routine: Callable[[float, U], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> Callable[P, WriterFactory]:
    ...

@overload
def writer(
    u: str,
    condition: int | float | None = None,
    routine: Callable[[float, U], None] | None = None,
    dir_path: str | None = None,
    file_name: str | None = None,
) -> WriterFactory:
    ...


def writer(
    u: str | U,
    condition: Callable[Concatenate[U, P], bool] | int | float |  None = None,
    routine: Callable[[float, U], None] | None = None,
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
                return Writer(_routine, create_lazy_evaluator(condition)(u, *args, **kwargs), u.name)
        return _


def as_writer(
    arg: Writer | LazyEvaluator[bool] | WriterFactory,
    sim: Simulation,
) -> Writer:
    if isinstance(arg, Writer):
        return arg
    elif is_controller_factory(arg):
        _writer = arg(sim)
        if not isinstance(_writer, Writer):
            raise TypeError(f'{_writer} is not a `Writer`.')
        return _writer
    else:
        return Writer(arg)
    

Controller: TypeAlias = Stopper | Writer
ControllerFactory: TypeAlias = StopperFactory | WriterFactory


def is_controller(
    obj: Controller | None,
):
    return isinstance(obj, Controller)


def is_controller_factory(
    clbl: Callable | ControllerFactory,
    variadic: bool = False,
) -> bool:
    """
    Returns `True` if the callable's first and only argument is of type `Simulation`.
    """
    if not callable(clbl):
        raise TypeError(f'{clbl} is not callable.')

    if not variadic and len(signature(clbl).parameters) != 1:
        return False
    annt = list(signature(clbl).parameters.values())[0].annotation

    if annt is Parameter.empty:
        raise TypeError(f'Type hint cannot be empty.')

    return list(signature(clbl).parameters.values())[0].annotation is Simulation