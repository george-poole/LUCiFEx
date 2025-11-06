from typing import (
    Callable, ParamSpec, Iterable, Generic, 
    TypeVar, Any, Literal,
)
from typing_extensions import Self
import numpy as np

from dolfinx.fem import Expression #Function, Constant,
from ufl import Measure
from ufl.core.expr import Expr

from ..utils import set_value, replicate_callable, SpatialMarkerAlias, as_dofs_setter
from ..fem import Constant, Function
from ..fdm.series import ConstantSeries, FunctionSeries
from .options import OptionsFFCX, OptionsJIT
from .bcs import SubspaceIndex


S = TypeVar('S')
T = TypeVar('T')
P = ParamSpec("P")
class Evaluation(Generic[S, T]):
    
    def __init__(
        self,
        solution: S | T ,
        evaluation: Callable[[], Any],
        future: bool = False,
        overwrite: bool = False,
    ) -> None:
        if isinstance(solution, Constant):
            uh = solution
            uxt = ConstantSeries(uh.mesh, uh.name, shape=uh.ufl_shape)
        elif isinstance(solution, ConstantSeries):
            uxt = solution
            uh = Constant(uxt.mesh, name=uxt.name, shape=uxt.shape)
        elif isinstance(solution, Function):
            uh = solution
            uxt = FunctionSeries(uh.function_space, uh.name)
        elif isinstance(solution, FunctionSeries):
            uxt = solution
            uh = Function(uxt.function_space, name=uxt.name)
        else:
            raise TypeError(f"{type(solution)}")

        self._series = uxt
        self._solution = uh
        self._evaluation = evaluation
        self._future = future
        self._overwrite = overwrite

    @classmethod
    def from_function(
        cls, 
        solution: S | T, 
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

    @property
    def series(self) -> S:
        return self._series

    @property
    def solution(self) -> T:
        return self._solution

    def solve(
        self, 
        future: bool | None = None, 
        overwrite: bool | None = None,
    ) -> None:
        if future is None:
            future = self._future
        if overwrite is None:
            overwrite = self._overwrite
        set_value(self._solution, self._evaluation())
        self._series.update(self._solution, future, overwrite)

    def forward(self, t: float | np.ndarray | Constant):
        self._series.forward(t)


P = ParamSpec("P")
class Interpolation(Evaluation[FunctionSeries, Function]):

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
        dofs_corrector: Callable[[Function], None] 
        | Iterable[tuple[SpatialMarkerAlias, float | Constant] | tuple[SpatialMarkerAlias, float | Constant, SubspaceIndex]] 
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

        _f = Function(solution.function_space)
        dofs_corrector = as_dofs_setter(dofs_corrector)
        def evaluation() -> Function:
            _f.interpolate(expr)
            dofs_corrector(_f)
            return _f

        super().__init__(solution, evaluation, future, overwrite)

    @classmethod
    def from_function(
        cls,
        solution: Function | FunctionSeries, 
        expression_func: Callable[P, Function | Expr],
        dofs_corrector: Callable[[Function], None] 
        | Iterable[tuple[SpatialMarkerAlias, float | Constant] | tuple[SpatialMarkerAlias, float | Constant, SubspaceIndex]] 
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
                dofs_corrector, 
                jit, 
                ffcx, 
                future,
                overwrite,
            )
        return _create


P = ParamSpec("P")
class Integration(Evaluation[ConstantSeries, Constant]):
    
    @classmethod
    def from_function(
        cls, 
        solution: ConstantSeries | Constant, 
        func: Callable[P, Expr | tuple[Expr, ...]],
        measure: Literal['dx', 'ds', 'dS'] | Measure | None = None, 
        *markers: SpatialMarkerAlias,
        future: bool = False,
        overwrite: bool = False,
        facet_side: Literal['+', '-'] | None = None,
        **metadata,
    ):
        if measure is None:
            _func = func
        else:
            _func = func(
            measure,
            *markers,
            facet_side=facet_side,
            **metadata,
            )

        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(solution, lambda: _func(*args, **kwargs), future, overwrite)
        
        return _create
    

@replicate_callable(Evaluation[ConstantSeries | FunctionSeries, Constant | Function].from_function)
def evaluation():
    pass

@replicate_callable(Interpolation.from_function)
def interpolation():
    pass

@replicate_callable(Integration.from_function)
def integration():
    pass


