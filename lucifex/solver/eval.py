from abc import ABC, abstractmethod
from typing import (
    Callable, ParamSpec, Generic,
    TypeVar, Any, Literal,
)
from typing_extensions import Self
import numpy as np

from dolfinx.fem import Expression #Function, Constant,
from ufl import Measure
from ufl.core.expr import Expr

from ..utils import replicate_callable, SpatialMarkerAlias, MultipleDispatchTypeError
from ..fem import Constant, Function
from ..fdm.series import ConstantSeries, FunctionSeries
from .options import OptionsFFCX, OptionsJIT


T = TypeVar('T', Function, Constant)
TS = TypeVar('TS', FunctionSeries, ConstantSeries)
class GenericSolver(ABC, Generic[T, TS]):

    def __init__(
        self,
        solution: T | TS | Any, #FIXME typing
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        future: bool = False,
        overwrite: bool = False,
        ):
        if isinstance(solution, (Function, Constant)):
            series = None
        elif isinstance(solution, FunctionSeries):
            series = solution
            solution = Function(series.function_space, name=series.name)
        elif isinstance(solution, ConstantSeries):
            series = solution
            solution = Constant(series.mesh, name=series.name, shape=series.shape)
        else:
            raise MultipleDispatchTypeError(solution)

        self._solution = solution
        self._series = series
        self._future = future
        self._overwrite = overwrite

        self._corrector_func = None
        self._correction = None
        self._correction_series = None
        if callable(corrector):
            self._corrector_func = corrector
        if isinstance(corrector, tuple):
            corr_name, self._corrector_func = corrector
            if corr_name == self._solution.name:
                raise ValueError('Solution and correction names must be distinct.')
            if isinstance(self._solution, Function):
                self._correction = Function(self._solution.function_space, name=corr_name)
                if self._series is not None:
                    self._correction_series = FunctionSeries(
                        self._solution.function_space, corr_name, series.order, series.store,
                    )
            elif isinstance(self._solution, Constant):
                self._correction = Constant(
                    self._solution.mesh, name=corr_name, shape=self._solution.ufl_shape,
                )
                if self._series is not None:
                    self._correction_series = ConstantSeries(
                        self._solution.mesh, corr_name, self._series.order, self._series.shape, self._series.store,
                    )
            else:
                raise TypeError
            
    @property
    def solution(self) -> T:
        return self._solution

    @property
    def series(self) -> TS | None:
        return self._series
    
    @property
    def correction(self) -> T | None:
        return self._correction
    
    @property
    def correction_series(self) -> TS | None:
        return self._correction_series
    
    @abstractmethod
    def solve(
        self, 
        future: bool | None,
        overwrite: bool | None,
    ) -> None:
        """
        In-place mutation of `self._solution`
        """
        if future is None:
            future = self._future
        if overwrite is None:
            overwrite = self._overwrite

        if self._corrector_func is not None:
            if self._correction is not None:
                self._correction.x.array[:] = self._solution.x.array[:]
            self._corrector_func(self._solution.x.array)
            if self._correction is not None:
                self._correction.x.array[:] = self._solution.x.array[:] - self._correction.x.array[:]
            if self._correction_series is not None:
                self._correction_series.update(self._correction, future, overwrite)

        if self._series is not None:
            self._series.update(self._solution, future, overwrite)


P = ParamSpec("P")
class Evaluation(GenericSolver[T, TS]):
    
    def __init__(
        self,
        solution: T | TS ,
        evaluation: Callable[[], Any],
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        future: bool = False,
        overwrite: bool = False,
    ) -> None:
        super().__init__(solution, corrector, future, overwrite)
        self._evaluation = evaluation

    @classmethod
    def from_function(
        cls, 
        solution: T | TS, 
        func: Callable[P, Any],
        future: bool = False,
        overwrite: bool = False,
    ):
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(solution, lambda: func(*args, **kwargs), future, overwrite)
        return _create

    def solve(
        self, 
        future: bool | None = None, 
        overwrite: bool | None = None,
    ) -> None:
        self._series.set_container(self._solution, self._evaluation())
        super().solve(future, overwrite)


P = ParamSpec("P")
class Interpolation(Evaluation[Function, FunctionSeries]):

    jit_default = OptionsJIT.default()
    ffcx_default = OptionsFFCX.default()

    @classmethod
    def set_defaults(
        cls,
        jit=None,
        ffcx=None,
    ):
        if jit is None:
            jit = OptionsJIT.default()
        if ffcx is None:
            ffcx = OptionsFFCX.default()
        cls.jit_default = jit
        cls.ffcx_default = ffcx

    def __init__(
        self,
        solution: Function | FunctionSeries,
        expr: Expr | Function,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        future: bool = False,
        overwrite: bool = False,
    ):
        if jit is None:
            jit = self.jit_default
        if ffcx is None:
            ffcx = self.ffcx_default
        jit = dict(jit)
        ffcx = dict(ffcx)

        if isinstance(expr, Expr):
            expr = Expression(
                expr,
                solution.function_space.element.interpolation_points(),
                ffcx,
                jit,
            )

        _solution = Function(solution.function_space)
        def _evaluation() -> Function:
            _solution.interpolate(expr)
            return _solution

        super().__init__(solution, _evaluation, corrector, future, overwrite)

    @classmethod
    def from_function(
        cls,
        solution: Function | FunctionSeries, 
        expression_func: Callable[P, Function | Expr],
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        future: bool = False,
        overwrite: bool = False,
    ):
        """from function"""
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(
                solution, 
                expression_func(*args, **kwargs), 
                corrector, 
                jit, 
                ffcx, 
                future,
                overwrite,
            )
        return _create
    

P = ParamSpec("P")
class Integration(Evaluation[Constant, ConstantSeries]):
    
    @classmethod
    def from_function(
        cls, 
        solution: Constant | ConstantSeries, 
        integrand_func: Callable[P, Expr | tuple[Expr, ...]],
        measure: Literal['dx', 'ds', 'dS'] | Measure | None = None, 
        *markers: SpatialMarkerAlias,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        future: bool = False,
        overwrite: bool = False,
        facet_side: Literal['+', '-'] | None = None,
        **measure_kwargs,
    ):
        if measure is None:
            _func = integrand_func
        else:
            _func = integrand_func(
            measure,
            *markers,
            facet_side=facet_side,
            **measure_kwargs,
            )

        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(solution, lambda: _func(*args, **kwargs), corrector, future, overwrite)
        
        return _create
    

@replicate_callable(Evaluation[Constant | Function, ConstantSeries | FunctionSeries].from_function)
def evaluation():
    pass

@replicate_callable(Interpolation.from_function)
def interpolation():
    pass

@replicate_callable(Integration.from_function)
def integration():
    pass