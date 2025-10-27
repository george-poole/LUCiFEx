from typing import (
    Callable, ParamSpec, Iterable, Generic, 
    TypeVar, Any, Literal,
)
from typing_extensions import Self
import numpy as np

from dolfinx.fem import Function, Constant, Expression
from ufl import Measure
from ufl.core.expr import Expr

from ..utils import set_value, replicate_callable, Marker, SubspaceIndex, as_dofs_setter
from ..fem import SpatialConstant, SpatialFunction
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
        if isinstance(solution, SpatialConstant):
            uh = solution
            uxt = ConstantSeries(uh.mesh, uh.name, shape=uh.ufl_shape)
        elif isinstance(solution, ConstantSeries):
            uxt = solution
            uh = SpatialConstant(uxt.mesh, name=uxt.name, shape=uxt.shape)
        elif isinstance(solution, SpatialFunction):
            uh = solution
            uxt = FunctionSeries(uh.function_space, uh.name)
        elif isinstance(solution, FunctionSeries):
            uxt = solution
            uh = SpatialFunction(uxt.function_space, name=uxt.name)
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
class Interpolation(Evaluation[FunctionSeries, SpatialFunction]):

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
        solution: SpatialFunction | FunctionSeries,
        expression: SpatialFunction | Expr,
        dofs_corrector: Callable[[SpatialFunction], None] 
        | Iterable[tuple[Marker, float | SpatialConstant] | tuple[Marker, float | SpatialConstant, SubspaceIndex]] 
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

        if isinstance(expression, Expr):
            expression = Expression(
                expression,
                solution.function_space.element.interpolation_points(),
                ffcx,
                jit,
            )

        _f = SpatialFunction(solution.function_space)
        dofs_corrector = as_dofs_setter(dofs_corrector)
        def evaluation() -> SpatialFunction:
            _f.interpolate(expression)
            dofs_corrector(_f)
            return _f

        super().__init__(solution, evaluation, future, overwrite)

    @classmethod
    def from_function(
        cls,
        solution: Function | FunctionSeries, 
        expression_func: Callable[P, SpatialFunction | Expr],
        dofs_corrector: Callable[[SpatialFunction], None] 
        | Iterable[tuple[Marker, float | SpatialConstant] | tuple[Marker, float | SpatialConstant, SubspaceIndex]] 
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
class Integration(Evaluation[ConstantSeries | FunctionSeries, SpatialConstant | SpatialFunction]):
    
    @classmethod
    def from_function(
        cls, 
        solution: S | T, 
        func: Callable[P, Expr | tuple[Expr, ...]],
        measure: Literal['dx', 'ds', 'dS'] | Measure | None = None, 
        *markers: Marker,
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
    

@replicate_callable(Evaluation[ConstantSeries | FunctionSeries, SpatialConstant | SpatialFunction].from_function)
def evaluation():
    pass

@replicate_callable(Interpolation.from_function)
def interpolation():
    pass

@replicate_callable(Integration.from_function)
def integration():
    pass



# class CellIntegrationProblem(IntegrationProblem):
#     _measure_type = 'dx'

    # @classmethod
    # def from_function(
    #     cls, 
    #     solution: ConstantSeries | SpatialConstant,
    #     integrand_func: Callable[P, Expr | tuple[Expr, ...] | Form],
    #     marker: SpatialMarkerTypes | Measure | None = None,
    #     quadrature_degree: int | None = None, 
    #     future: bool = False,
    # ):
    #     def _create(
    #         *args: P.args, 
    #         **kwargs: P.kwargs,
    #     ) -> Self:
    #         return cls(solution, integrand_func(*args, **kwargs), marker, quadrature_degree, future)
    #     return _create


# class FacetIntegrationProblem(IntegrationProblem):
#     _measure_type = 'ds'


# P = ParamSpec("P")
# class InteriorFacetIntegrationProblem(IntegrationProblem):
#     _measure_type = 'dS'

#     def __init__(
#         self,
#         solution: ConstantSeries | SpatialConstant,
#         integrand: Expr | tuple[Expr, ...] | Form,
#         marker: SpatialMarkerTypes | Measure | None = None, 
#         quadrature_degree: int | None = None, 
#         facet_side: Literal['+', '-'] = '+',
#         future: bool = False,
#     ):
#         if isinstance(integrand, tuple):
#             integrand = tuple(i(facet_side) if not isinstance(i, Restricted) else i for i in integrand)

#         if isinstance(integrand, Expr) and not isinstance(integrand, Restricted):
#             integrand = integrand(facet_side)

#         super().__init__(solution, integrand, marker, quadrature_degree, future)

#     @classmethod
#     def from_function(
#         cls, 
#         solution: ConstantSeries | SpatialConstant,
#         integrand_func: Callable[P, Expr | tuple[Expr, ...] | Form], 
#         marker: SpatialMarkerTypes | Measure | None = None,
#         quadrature_degree: int | None = None, 
#         facet_side: Literal['+', '-'] = '+',
#         future: bool = False,
#     ):
#         def _create(
#             *args: P.args, 
#             **kwargs: P.kwargs,
#         ) -> Self:
#             return cls(solution, integrand_func(*args, **kwargs), marker, quadrature_degree, facet_side, future)
#         return _create





# @replicate_callable(FacetIntegrationProblem.from_function)
# def ds_solver():
#     pass

# @replicate_callable(InteriorFacetIntegrationProblem.from_function)
# def dS_solver():
#     pass


# _measure_type: str 

# def __init__(
#     self,
#     solution: ConstantSeries | SpatialConstant,
#     ###
#     integrand: Expr | tuple[Expr, ...] | Form,
#     marker: SpatialMarkerTypes | Measure | None = None,
#     quadrature_degree: int | None = None,  # TODO expand to quadrature metadata/scheme
#     ###
#     future: bool = False,
# ):
#     if isinstance(integrand, Form):
#         if marker is not None:
#             raise ValueError('Integration measure is already specified in the `Form` object supplied.')
#         dx = None # TODO get Measure from Form somehow?
#         evaluation = lambda: assemble_scalar(form(integrand))
#     else:
#         ...







#         #####
#         if marker is None:
#             dx = create_tagged_measure(self._measure_type, solution.mesh)(degree=quadrature_degree)
#             evaluation = lambda: assemble_scalar(form(integrand * dx))
#         elif isinstance(marker, Measure):
#             dx = marker(degree=quadrature_degree)
#             evaluation = lambda: assemble_scalar(form(integrand * dx))
#         elif isinstance(marker, (list, tuple)):
#             tags = list(range(len(marker)))
#             dx = lambda tag: create_tagged_measure(
#                 self._measure_type, 
#                 solution.mesh, 
#                 marker,
#                 tags,
#             )(tag, degree=quadrature_degree)
#             if isinstance(integrand, tuple):
#                 assert solution.shape == (len(marker), len(integrand))
#                 evaluation = lambda: np.array([[assemble_scalar(form(i * dx(tag))) for i in integrand] for tag in tags])
#             else:   
#                 assert solution.shape == (len(marker), )
#                 evaluation = lambda: tuple(assemble_scalar(form(integrand * dx(t)) for t in tags))
#         else:
#             tag = 0
#             dx = create_tagged_measure(self._measure_type, 
#                                             solution.mesh, 
#                                             [marker], [tag],
#                                             )(tag, degree=quadrature_degree)
#             if isinstance(integrand, tuple):
#                 assert solution.shape == (len(integrand), )
#                 evaluation = lambda: tuple(assemble_scalar(form(i * dx)) for i in integrand)
#             else:   
#                 evaluation = lambda: assemble_scalar(form(integrand * dx))
#         ####

#     self._measure = dx
#     super().__init__(solution, evaluation, future)
