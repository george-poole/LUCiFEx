from typing import Iterable, Any, Callable, overload
from typing_extensions import Self

from ufl.core.expr import Expr
from ufl import TrialFunction
from dolfinx.fem import Function, Constant

from .series import FunctionSeries, ConstantSeries, ExprSeries, Series


class FiniteDifference:

    trial_default = True

    def __init__(
        self,
        indices_coeffs: dict[int, float] | tuple[Iterable[int], Iterable[float]],
        init: 
            tuple[Iterable[int], Iterable[float]] | dict[int, float] | Self | None
        = None,
        name: str | None = None,
    ) -> None:

        if not isinstance(indices_coeffs, dict):
            indices_coeffs = dict(zip(*indices_coeffs, strict=True))

        self._coefficients = indices_coeffs

        if self.order > 1:
            if init is None:
                raise ValueError(
                    "Must also provide first order method for initialization"
                )

        if isinstance(init, FiniteDifference):
            assert init.order in (0, 1)
            self._init = init
        elif isinstance(init, dict):
            self._init = self(init)
        elif isinstance(init, tuple):
            self._init = FiniteDifference(*init, init=None)
        elif init is None:
            assert self.order in (0, 1)
            self._init = init
        else:
            raise TypeError
        
        if name is None:
            name = f'FD{id(self)}'
        self._name = name

        self.trial = self.trial_default
    
    @property
    def coefficients(self) -> dict[int, float]:
        return self._coefficients
    
    @property
    def explicit_coeff(self) -> float:
        return self.coefficients.get(Series.FUTURE_INDEX, 0.0)

    @property
    def order(self) -> int:
        return Series.FUTURE_INDEX - min(self.coefficients)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def __name__(self) -> str: 
        # mimicking cls.__name__
        return self.name

    @property
    def init(self) -> Self | None:
        return self._init

    @property
    def is_implicit(self) -> bool:
        return max(self.coefficients) == Series.FUTURE_INDEX

    @property
    def is_explicit(self) -> bool:
        return not self.is_implicit

    def __call__(
        self,
        u: Series,
        trial: bool | None = None,
    ) -> Expr:
        if not isinstance(u, Series):
            raise TypeError(f"Expected argument of type {Series}, not {type(u)}.")
        
        if trial is None:
            trial = self.trial if isinstance(u, FunctionSeries) else trial

        if trial:
            assert isinstance(u, FunctionSeries)
            _u = lambda n: u[n] if n != u.FUTURE_INDEX else TrialFunction(u.function_space)
        else:
            _u = lambda n: u[n]

        return sum((c * _u(n) for n, c in self.coefficients.items()))

    def __repr__(self) -> str:
        return self.name
    

class FiniteDifferenceDerivative(FiniteDifference):

    def __init__(
        self,
        denominator: Callable[[Constant], Constant | Expr] | Callable[[float], float],
        indices_coeffs: dict[int, float] | tuple[Iterable[int], Iterable[float]],
        init: 
            tuple[Iterable[int], Iterable[float]] | dict[int, float] | Self | None
        = None,
        name: str | None = None,
    ) -> None:
        super().__init__(indices_coeffs, init, name)
        self._denominator = denominator

    def __call__(
        self, 
        u: FunctionSeries,
        dt: ConstantSeries | Constant | None = None,
        trial: bool | None = None,
    ) -> Expr:
        du = super().__call__(u, trial)
        if dt is None:
            return du
        if isinstance(dt, ConstantSeries):
            _dt = dt[0]
        else:
            _dt = dt
        return du / self._denominator(_dt)
    
    @property
    def denominator(self) -> Callable[[Constant], Constant | Expr] | Callable[[float], float]:
        return self._denominator


CN = FiniteDifference(
    {
        Series.FUTURE_INDEX: 0.5,
        Series.FUTURE_INDEX - 1: 0.5,
    },
    name='CN'
)
"""Crank-Nicolson"""

FE = FiniteDifference({Series.FUTURE_INDEX - 1: 1.0}, name='FE')
"""Forward Euler explicit method"""

BE = FiniteDifference({Series.FUTURE_INDEX: 1.0}, name='BE')
"""Backward Euler implicit method"""

DT = FiniteDifferenceDerivative(
    lambda dt: dt,
    {
        Series.FUTURE_INDEX: 1.0,
        Series.FUTURE_INDEX - 1: -1.0,
    },
    name='DT'
)
"""Forward first time derivative"""

DTLF = FiniteDifferenceDerivative(
    lambda dt: dt,
    {
        Series.FUTURE_INDEX: 0.5,
        Series.FUTURE_INDEX - 2: -0.5,
    },
    DT,
    name='LF'
)
"""Centred (leapfrog) first time derivative"""

DT2 = FiniteDifferenceDerivative(
    lambda dt: dt ** 2,
    {
        Series.FUTURE_INDEX: 1.0,
        Series.FUTURE_INDEX - 1: -2.0,
        Series.FUTURE_INDEX - 2: 1.0,
    },
    DT,
    name='DT2'
)
"""Centred second time derivative"""


def AB(n: int, init: FiniteDifference | None = None) -> FiniteDifference:
    """Adams-Bashforth family of explicit methods"""
    match n:
        case 1:
            d = FE.coefficients
        case 2:
            d = {
                Series.FUTURE_INDEX - 1: 1.5,
                Series.FUTURE_INDEX - 2: -0.5,
            }
        case 3:
            d = {
                Series.FUTURE_INDEX - 1: 23 / 12,
                Series.FUTURE_INDEX - 2: -16 / 12,
                Series.FUTURE_INDEX - 3: 5 / 12,
            }
        case 4:
            d = {
                Series.FUTURE_INDEX: 55 / 24,
                Series.FUTURE_INDEX - 1: -59 / 24,
                Series.FUTURE_INDEX - 2: 37 / 24,
                Series.FUTURE_INDEX - 3: -9 / 24,
            }
        case _:
            raise NotImplementedError(f'Only implemented up to AB4.')

    if init is None:
        init = FE

    return FiniteDifference(d, init, f'AB{n}')


AB1 = AB(1)
AB2 = AB(2)
AB3 = AB(3)

def AM(
    n: int, 
    init: FiniteDifference | None = None,
) -> FiniteDifference:
    """Adams-Moulton family of implicit methods"""
    match n:
        case 1:
            d = BE.coefficients
        case 2:
            d = {
                Series.FUTURE_INDEX: 0.5,
                Series.FUTURE_INDEX - 1: 0.5,
            }
        case 3:
            d = {
                Series.FUTURE_INDEX: 5 / 12,
                Series.FUTURE_INDEX - 1: 8 / 12,
                Series.FUTURE_INDEX - 2: -1 / 12,
            }
        case 4:
            d = {
                Series.FUTURE_INDEX: 9 / 24,
                Series.FUTURE_INDEX - 1: -9 / 24,
                Series.FUTURE_INDEX - 2: -5 / 24,
                Series.FUTURE_INDEX - 3: 1 / 24,
            }
        case _:
            raise NotImplementedError(f'Only implemented up to AM4.')

    if init is None:
        init = BE

    return FiniteDifference(d, init, f'AM{n}')


AM1 = AM(1)
AM2 = AM(2)
AM3 = AM(3)


def BDF(
    n: int,
    init: FiniteDifference | None = None,
) -> FiniteDifferenceDerivative:
    """Backward-differentiation-formulae family for discretizations of `∂u/∂t`"""
    match n:
        case 1:
            d = DT.coefficients
        case 2:
            d = (
                {
                    Series.FUTURE_INDEX: 1.5,
                    Series.FUTURE_INDEX - 1: -2.0,
                    Series.FUTURE_INDEX - 2: 0.5,
                }
            )
        case 3:
            d = {
                Series.FUTURE_INDEX: 11 / 6,
                Series.FUTURE_INDEX - 1: -3.0,
                Series.FUTURE_INDEX - 2: 1.5,
                Series.FUTURE_INDEX - 3: -1 / 3,
            }
        case 4:
            d = {
                Series.FUTURE_INDEX: 25 / 12,
                Series.FUTURE_INDEX - 1: -4.0,
                Series.FUTURE_INDEX - 2: 3.0,
                Series.FUTURE_INDEX - 3: -4 / 3,
                Series.FUTURE_INDEX - 4: 1 / 4,
            }
        case _:
            raise NotImplementedError(f'Only implemented up to BDF4.')

    if init is None:
        init = DT

    return FiniteDifferenceDerivative(lambda dt: dt, d, init, f'BDF{n}')


BDF1 = BDF(1)
BDF2 = BDF(2)
BDF3 = BDF(3)


def THETA(theta: float) -> FiniteDifference:
    """
    `θuⁿ⁺¹ + (1 - θ)uⁿ`

    `θ = 0` forward Euler \\
    `θ = 0.5` Crank-Nicolson \\
    `θ = 1` backward Euler
    """
    return FiniteDifference(
        {
            Series.FUTURE_INDEX: theta, 
            Series.FUTURE_INDEX - 1: 1 - theta,
        }, 
        name=f'THETA{theta}',
    )


def finite_difference_order(*args: FiniteDifference | tuple[FiniteDifference, ...] | Any) -> int | None:

    args_fd = [a for a in args if isinstance(a, FiniteDifference)]
    for a in args:
        if isinstance(a, tuple) and all(isinstance(i, FiniteDifference) for i in a):
            args_fd.extend(a)

    orders = [a.order for a in args_fd]
    if orders:
        return max(1, max(orders))
    else:
        return None
    

def finite_difference_argwise(
    D_fdm: tuple[FiniteDifference, ...],
    expr: ExprSeries | tuple[Callable, tuple], 
    trial: FunctionSeries | None = None,
) -> Expr:
    if isinstance(expr, tuple):
        expr_func, expr_args = expr
    else:
        assert isinstance(expr, ExprSeries)
        assert expr.relation
        expr_func, expr_args = expr.relation
    use_trial = [arg is trial for arg in expr_args]
    return expr_func(
        *(Di(arg, ut) for Di, arg, ut in zip(D_fdm, expr_args, use_trial, strict=False)),
    )
    

@overload
def apply_finite_difference(
    D_fdm: FiniteDifference | tuple[FiniteDifference, ...],
    expr: Function | Expr | Constant, 
) -> Function | Expr | Constant:
    ...


@overload
def apply_finite_difference(
    D_fdm: FiniteDifference,
    expr: Series, 
    trial: bool | None = None,
) -> Expr:
    ...


@overload
def apply_finite_difference(
    D_fdm: tuple[FiniteDifference, ...],
    expr: ExprSeries | tuple[Callable, tuple], 
    trial: FunctionSeries | bool | None = None,
) -> Expr:
    ...
    

def apply_finite_difference(
    D_fdm: FiniteDifference | tuple[FiniteDifference, ...],
    expr: Function 
    | Expr
    | Constant 
    | Series
    | tuple[Callable[[tuple[Series, ...]], Expr], tuple[Series, ...]],
    trial: FunctionSeries | bool | None = None,
) -> Expr:
    if isinstance(expr, (Function, Expr, Constant)):
        return expr
    if isinstance(D_fdm, FiniteDifference):
        return D_fdm(expr, trial)
    if isinstance(D_fdm, tuple):
        return finite_difference_argwise(D_fdm, expr, trial)