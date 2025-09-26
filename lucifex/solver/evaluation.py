from typing import Callable, ParamSpec, Iterable, Generic, TypeVar, Any, Literal
from typing_extensions import Self
import numpy as np

from dolfinx.fem import Function, Constant, assemble_scalar, Expression
from ufl import Form, Measure, TestFunction, TrialFunction, inner
from ufl.restriction import Restricted
from ufl.core.expr import Expr

from ..fdm.ufl_operators import inner
from ..utils import set_value, copy_callable, SpatialMarker, as_dofs_setter
from ..fem import LUCiFExConstant, LUCiFExFunction
from ..fdm.series import ConstantSeries, FunctionSeries

from .petsc import form
from .pde import BoundaryValueProblem, OptionsPETSc, OptionsFFCX, OptionsJIT
from .bcs import create_enumerated_measure, BoundaryConditions, Value, SubspaceIndex


S = TypeVar('S')
T = TypeVar('T')
P = ParamSpec("P")
class EvaluationProblem(Generic[S, T]):
    
    def __init__(
        self,
        solution: S | T,
        evaluation: Callable[[], Any],
        future: bool = False,
    ) -> None:
        if isinstance(solution, LUCiFExConstant):
            uh = solution
            uxt = ConstantSeries(uh.mesh, uh.name, shape=uh.ufl_shape)
        elif isinstance(solution, ConstantSeries):
            uxt = solution
            uh = LUCiFExConstant(uxt.mesh, name=uxt.name, shape=uxt.shape)
        elif isinstance(solution, LUCiFExFunction):
            uh = solution
            uxt = FunctionSeries(uh.function_space, uh.name)
        elif isinstance(solution, FunctionSeries):
            uxt = solution
            uh = LUCiFExFunction(uxt.function_space, name=uxt.name)
        else:
            raise TypeError(f"{type(solution)}")

        self._series = uxt
        self._solution = uh
        self._evaluation = evaluation
        self._future = future

    @classmethod
    def from_function(
        cls, 
        solution: S | T, 
        func: Callable[P, Any],
    ):
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(solution, lambda: func(*args, **kwargs))
        return _create

    @property
    def series(self) -> S:
        return self._series

    @property
    def solution(self) -> T:
        return self._solution

    def solve(self, future: bool | None = None, overwrite: bool = False) -> None:
        if future is None:
            future = self._future
        set_value(self._solution, self._evaluation())
        self._series.update(self._solution, future, overwrite)

    def forward(self, t: float | np.ndarray | Constant):
        self._series.forward(t)


P = ParamSpec("P")
class IntegrationProblem(EvaluationProblem[ConstantSeries | FunctionSeries, LUCiFExConstant | LUCiFExFunction]):

    _measure_type: str 

    def __init__(
        self,
        solution: ConstantSeries | LUCiFExConstant,
        integrand: Expr | tuple[Expr, ...] | Form,
        marker: SpatialMarker | Iterable[SpatialMarker] | Measure | None = None, # TODO expand to broader SpatialMarker type which further supports `int | str` tagging
        quadrature_degree: int | None = None,  # TODO expand to quadrature metadata/scheme
    ):

        if isinstance(integrand, Form):
            if marker is not None:
                raise ValueError('Integration measure is already specified in the `Form` object supplied.')
            dx = None # TODO get Measure from Form somehow?
            evaluation = lambda: assemble_scalar(form(integrand))
        else:
            if marker is None:
                dx = create_enumerated_measure(self._measure_type, solution.mesh)(degree=quadrature_degree)
                evaluation = lambda: assemble_scalar(form(integrand * dx))
            elif isinstance(marker, Measure):
                dx = marker(degree=quadrature_degree)
                evaluation = lambda: assemble_scalar(form(integrand * dx))
            elif isinstance(marker, (list, tuple)): #FIXME Iterable or Sequence
                tags = list(range(len(marker)))
                dx = lambda tag: create_enumerated_measure(self._measure_type, 
                                                solution.mesh, 
                                                marker, tags,
                                                )(tag, degree=quadrature_degree)
                if isinstance(integrand, tuple):
                    assert solution.shape == (len(marker), len(integrand))
                    evaluation = lambda: np.array([[assemble_scalar(form(i * dx(tag))) for i in integrand] for tag in tags])
                else:   
                    assert solution.shape == (len(marker), )
                    evaluation = lambda: tuple(assemble_scalar(form(integrand * dx(t)) for t in tags))
            else:
                tag = 0
                dx = create_enumerated_measure(self._measure_type, 
                                                solution.mesh, 
                                                [marker], [tag],
                                                )(tag, degree=quadrature_degree)
                if isinstance(integrand, tuple):
                    assert solution.shape == (len(integrand), )
                    evaluation = lambda: tuple(assemble_scalar(form(i * dx)) for i in integrand)
                else:   
                    evaluation = lambda: assemble_scalar(form(integrand * dx))

        self._measure = dx
        super().__init__(solution, evaluation)
        # TODO parallel version
        # self._mesh.comm.gather(dfx.fem.assemble_scalar(self._compiled), root=0)
        # self._scalar = dfx.f.assemble_scalar(self._compiled)
        # self._mesh = ct.function_space.mesh

    @property
    def measure(self) -> Measure | None:
        return self._measure

    @classmethod
    def from_function(
        cls, 
        solution: ConstantSeries | LUCiFExConstant,
        integrand_func: Callable[P, Expr | tuple[Expr, ...] | Form],
        marker: SpatialMarker | Measure | None = None,
        quadrature_degree: int | None = None, 
    ):
        def _create(
            *args: P.args, 
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(solution, integrand_func(*args, **kwargs), marker, quadrature_degree)
        return _create
    

class CellIntegrationProblem(IntegrationProblem):
    _measure_type = 'dx'


class FacetIntegrationProblem(IntegrationProblem):
    _measure_type = 'ds'


P = ParamSpec("P")
class InteriorFacetIntegrationProblem(IntegrationProblem):
    _measure_type = 'dS'

    def __init__(
        self,
        solution: ConstantSeries | LUCiFExConstant,
        integrand: Expr | tuple[Expr, ...] | Form,
        marker: SpatialMarker | Measure | None = None, 
        quadrature_degree: int | None = None, 
        facet_side: Literal['+', '-'] = '+',
    ):
        if isinstance(integrand, tuple):
            integrand = tuple(i(facet_side) if not isinstance(i, Restricted) else i for i in integrand)

        if isinstance(integrand, Expr) and not isinstance(integrand, Restricted):
            integrand = integrand(facet_side)

        super().__init__(solution, integrand, marker, quadrature_degree)

    @classmethod
    def from_function(
        cls, 
        solution: ConstantSeries | LUCiFExConstant,
        integrand_func: Callable[P, Expr | tuple[Expr, ...] | Form], 
        marker: SpatialMarker | Measure | None = None,
        quadrature_degree: int | None = None, 
        facet_side: Literal['+', '-'] = '+',
    ):
        def _create(
            *args: P.args, 
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(solution, integrand_func(*args, **kwargs), marker, quadrature_degree, facet_side)
        return _create


P = ParamSpec("P")
class ProjectionProblem(BoundaryValueProblem):

    petsc_default = OptionsPETSc.default()
    jit_default = OptionsJIT.default()
    ffcx_default = OptionsFFCX.default()

    def __init__(
        self, 
        solution: LUCiFExFunction | FunctionSeries,
        expression: Function | Expr,
        bcs: BoundaryConditions | Iterable[tuple[SpatialMarker, Value] | tuple[SpatialMarker, Value, SubspaceIndex]] | None = None, 
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        dofs_corrector: Callable[[Function], None] | None = None,
    ):

        v = TestFunction(solution)
        u = TrialFunction(solution)
        dx = Measure('dx')
        F_lhs = inner(v, u) * dx
        F_rhs = inner(v, expression) * dx
        forms = [F_lhs, -F_rhs]

        if isinstance(bcs, Iterable):
            bcs = BoundaryConditions(*bcs)

        super().__init__(solution, forms, bcs, petsc, jit, ffcx, dofs_corrector, 
                         cache_matrix=True, use_partition=(False, False),)

    @classmethod
    def from_function(
        cls,
        solution: Function | FunctionSeries, 
        expression_func: Callable[P, Function | Expr],
        bcs: BoundaryConditions | Iterable[tuple[SpatialMarker, Value] | tuple[SpatialMarker, Value, SubspaceIndex]] | None = None, 
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        dofs_corrector: Callable[[Function], None] | None = None,
    ):
        """from function"""
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(solution, expression_func(*args, **kwargs), bcs, petsc, jit, ffcx, dofs_corrector)
        return _create
        

P = ParamSpec("P")
class InterpolationProblem(EvaluationProblem[FunctionSeries, LUCiFExFunction]):

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
        solution: LUCiFExFunction | FunctionSeries,
        expression: LUCiFExFunction | Expr,
        dofs_corrector: Callable[[LUCiFExFunction], None] 
        | Iterable[tuple[SpatialMarker, float | LUCiFExConstant] | tuple[SpatialMarker, float | LUCiFExConstant, SubspaceIndex]] 
        | None = None, 
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        future: bool = False,
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

        _f = LUCiFExFunction(solution.function_space)
        dofs_corrector = as_dofs_setter(dofs_corrector)
        def evaluation() -> LUCiFExFunction:
            _f.interpolate(expression)
            dofs_corrector(_f)
            return _f

        super().__init__(solution, evaluation, future)

    @classmethod
    def from_function(
        cls,
        solution: Function | FunctionSeries, 
        expression_func: Callable[P, LUCiFExFunction | Expr],
        dofs_corrector: Callable[[LUCiFExFunction], None] 
        | Iterable[tuple[SpatialMarker, float | LUCiFExConstant] | tuple[SpatialMarker, float | LUCiFExConstant, SubspaceIndex]] 
        | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        future: bool = False,
    ):
        """from function"""
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(solution, expression_func(*args, **kwargs), dofs_corrector, jit, ffcx, future)
        return _create


@copy_callable(EvaluationProblem[ConstantSeries | FunctionSeries, LUCiFExConstant | LUCiFExFunction].from_function)
def eval_solver():
    pass

@copy_callable(CellIntegrationProblem.from_function)
def dx_solver():
    pass

@copy_callable(FacetIntegrationProblem.from_function)
def ds_solver():
    pass

@copy_callable(InteriorFacetIntegrationProblem.from_function)
def dS_solver():
    pass

@copy_callable(ProjectionProblem.from_function)
def projection_solver():
    pass

@copy_callable(InterpolationProblem.from_function)
def interpolation_solver():
    pass
