from collections.abc import Callable
from inspect import Signature, signature
from typing import Concatenate, ParamSpec, overload, TypeVar, TypeAlias, Protocol, Generic

from dolfinx.fem import Constant, Function

from ..fdm import FunctionSeries, ConstantSeries
from ..io import write
from ..utils.deferred import Stopper, Writer
from .simulation import Simulation


T = TypeVar('T', FunctionSeries, ConstantSeries, Function, Constant)
P = ParamSpec('P')
StopperFromSimulation: TypeAlias = Callable[[Simulation], Stopper]


@overload
def stopper_from_series(
    u: str,
    condition: Callable[Concatenate[T, P], bool],
) -> Callable[P, StopperFromSimulation]:
    ...

@overload
def stopper_from_series(
    u: FunctionSeries | ConstantSeries,
    condition: Callable[Concatenate[T, P], bool],
) -> Callable[P, Stopper]:
    ...

def stopper_from_series(
    u: FunctionSeries | ConstantSeries | str,
    condition: Callable[Concatenate[T, P], bool],
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
            

T = TypeVar('T', FunctionSeries, ConstantSeries, Function, Constant)
P = ParamSpec('P')
WriterFromSimulation: TypeAlias = Callable[[Simulation], Writer]
R = TypeVar('R', Writer, WriterFromSimulation)

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
def writer_from_series(
    u: str,
) -> WriterFromStep[WriterFromSimulation]:
    ...

@overload
def writer_from_series(
    u: str,
    condition: Callable[Concatenate[T, P], bool],
) -> WriterFromParamspec[WriterFromSimulation, P]:
    ...

@overload
def writer_from_series(
    u: FunctionSeries,
) -> WriterFromStep[Writer]:
    ...


@overload
def writer_from_series(
    u: FunctionSeries,
    condition: Callable[Concatenate[T, P], bool],
) -> WriterFromParamspec[Writer, P]:
    ...


def writer_from_series(
    u: FunctionSeries | str,
    condition: Callable[Concatenate[T, P], bool] | None = None,
):

    def _inner(
        dir_path=None,
        file_name=None,
        *args,
        **kwargs,
        ):
        if isinstance(u, str):
            def _(sim: Simulation) -> Writer:
                if dir_path is None:
                    dir_path = sim.dir_path
                return writer_from_series(sim[u], condition)(dir_path, file_name, *args, **kwargs)
            return _
        else:
            if condition is not None:
                _condition = lambda: condition(*args, **kwargs)
            else:
                _condition = condition
            routine = lambda t: write(u, file_name, dir_path, t, 'a', u.mesh.comm)
            return Writer(routine, _condition, u.name)
            
    if condition is None:
        params = list(signature(WriterFromStep.__call__).parameters.values()[1:])
        _inner.__signature__ = Signature(params)
    else:
        params = [*list(signature(WriterFromStep.__call__).parameters.values()[1:3]),
                  *list(signature(condition).parameters.values())[1:],
                ]
        _inner.__signature__ = Signature(params)
    
    return _inner