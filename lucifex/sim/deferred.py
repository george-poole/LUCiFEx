from collections.abc import Callable
from inspect import Signature, signature
from typing import Concatenate, ParamSpec, overload, TypeVar, TypeAlias, Protocol, Generic, TypeAlias

from dolfinx.fem import Constant, Function

from ..fdm import FunctionSeries, ConstantSeries
from ..io import write
from ..utils.deferred import Stopper, Writer
from .simulation import Simulation


P = ParamSpec('P')
StopperFactory: TypeAlias = Callable[[Simulation], Stopper]

@overload
def create_stopper(
    u: FunctionSeries | ConstantSeries | Function | Constant,
    condition: Callable[Concatenate[FunctionSeries | ConstantSeries | Function | Constant, P], bool],
) -> Callable[P, Stopper]:
    ...

@overload
def create_stopper(
    u: str,
    condition: Callable[Concatenate[FunctionSeries | ConstantSeries | Function | Constant, P], bool],
) -> Callable[P, StopperFactory]:
    ...

def create_stopper(
    u: FunctionSeries | ConstantSeries | str,
    condition: Callable[..., bool],
):  
    paramspec_params = list(signature(condition).parameters.values())[1:]

    def _inner(*args: P.args, **kwargs: P.kwargs):
        if isinstance(u, str):
            _inner.__signature__ = Signature(paramspec_params, return_annotation=Callable[[Simulation], Stopper])
            _lambda = lambda sim: Stopper.from_args(condition)(sim[u], *args, **kwargs)
            return _lambda
        else:
            _inner.__signature__ = Signature(paramspec_params, return_annotation=Stopper)
            return Stopper.from_args(condition)(u, *args, **kwargs)

    return _inner
            

def as_stopper(
    arg: Stopper | Callable[[], bool] | StopperFactory,
    sim: Simulation,
) -> Stopper:
    if isinstance(arg, Stopper):
        return arg
    elif is_factory(arg):
        return arg(sim)
    else:
        return Stopper(arg)


T = TypeVar('T', FunctionSeries, ConstantSeries, Function, Constant)
P = ParamSpec('P')
WriterFactory: TypeAlias = Callable[[Simulation], Writer]
R = TypeVar('R', Writer, WriterFactory)

class WriterFromStep(Generic[R], Protocol):
    def __call__(
        self, 
        dir_path: str | None = None,
        file_name: str | None = None,
        step: int | float | None = None
    ) -> R:
        ...


class WriterFromParamspec(Generic[R, P],Protocol):
    def __call__(
        self, 
        dir_path: str | None = None,
        file_name: str | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        ...


@overload
def create_writer(
    u: FunctionSeries | ConstantSeries | Function | Constant,
    *,
    routine: Callable[[float, FunctionSeries | ConstantSeries], None] | None = None,
) -> WriterFromStep[Writer]:
    ...


@overload
def create_writer(
    u: FunctionSeries | ConstantSeries | Function | Constant,
    condition: Callable[Concatenate[FunctionSeries | ConstantSeries | Function | Constant, P], bool],
    *,
    routine: Callable[[float, FunctionSeries | ConstantSeries], None] | None = None,
) -> WriterFromParamspec[Writer, P]:
    ...

@overload
def create_writer(
    u: str,
    *,
    routine: Callable[[float, FunctionSeries | ConstantSeries], None] | None = None,
) -> WriterFromStep[WriterFactory]:
    ...

@overload
def create_writer(
    u: str,
    condition: Callable[Concatenate[FunctionSeries | ConstantSeries | Function | Constant, P], bool],
    *,
    routine: Callable[[float, FunctionSeries | ConstantSeries], None] | None = None,
) -> WriterFromParamspec[WriterFactory, P]:
    ...

def create_writer(
    u: FunctionSeries | ConstantSeries | Function | Constant | str,
    condition: Callable[..., bool] | None = None,
    *,
    routine: Callable[..., None] | None = None,
):

    def _inner(
        dir_path=None,
        file_name=None,
        *args,
        **kwargs,
        ):
        nonlocal routine
        if isinstance(u, str):
            def _(sim: Simulation) -> Writer:
                if dir_path is None:
                    dir_path = sim.dir_path
                return create_writer(sim[u], condition, routine=routine)(dir_path, file_name, *args, **kwargs)
            return _
        else:
            if condition is not None:
                _condition = lambda: condition(*args, **kwargs)
            else:
                _condition = condition
            if routine is None:
                routine = lambda t: write(u, file_name, dir_path, t, 'a', u.mesh.comm)
            else:
                routine = lambda t: routine(t, u)
            return Writer(routine, _condition, u.name)

    # FIXME      
    # if condition is None:
    #     params = list(signature(WriterFromStep.__call__).parameters.values())[1:]
    #     _inner.__signature__ = Signature(params)
    # else:
    #     params = [*list(signature(WriterFromStep.__call__).parameters.values())[1:3],
    #               *list(signature(condition).parameters.values())[1:],
    #             ]
    #     _inner.__signature__ = Signature(params)
    
    return _inner


def as_writer(
    arg: Writer | Callable[[], bool] | WriterFactory,
    sim: Simulation,
) -> Writer:
    if isinstance(arg, Writer):
        return arg
    elif is_factory(arg):
        return arg(sim)
    else:
        return Writer(arg)
    

def is_factory(func: Callable) -> bool:
    assert callable(func)
    if len(signature(func).parameters) != 1:
        return False
    return list(signature(func).parameters.values())[0].annotation is Simulation