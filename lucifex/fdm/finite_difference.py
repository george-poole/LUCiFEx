from typing import Iterable, Any, Callable
from typing_extensions import Self

from ufl.core.expr import Expr
from dolfinx.fem import Function, Constant

from .series import FunctionSeries, ConstantSeries, ExprSeries, Series

class FiniteDifference:
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
        
        if name is not None:
            self._name = name
        else:
            self._name = self.__class__.__name__

        self._trial = True
    
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
    
    # for compatibility with `eval`
    @property
    def __name__(self) -> str: 
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
    ) -> Expr:
        if not isinstance(u, Series):
            raise TypeError(f"Expected argument of type {Series}, not {type(u)}.")

        trial = self._trial if isinstance(u, FunctionSeries) else False

        if trial:
            _u = lambda n: u[n] if n != u.FUTURE_INDEX else u.trialfunction
        else:
            _u = lambda n: u[n]

        return sum((c * _u(n) for n, c in self.coefficients.items()))
    
    def __getitem__(
        self,
        trial: bool,
    ) -> Self:
        obj = FiniteDifference(self.coefficients, self.init, self.__name__)
        obj._trial = trial
        return obj

    def __repr__(self) -> str:
        return self.__name__
    

class FiniteDifferenceDerivative(FiniteDifference):

    def __init__(
        self,
        dt_denominator: Callable[[Constant], Constant | Expr] | Callable[[float], float],
        indices_coeffs: dict[int, float] | tuple[Iterable[int], Iterable[float]],
        init: 
            tuple[Iterable[int], Iterable[float]] | dict[int, float] | Self | None
        = None,
        name: str | None = None,
    ) -> None:
        super().__init__(indices_coeffs, init, name)
        self._dt_denominator = dt_denominator

    def __call__(
        self, 
        u: FunctionSeries,
        dt: ConstantSeries | Constant,
    ) -> Expr:
        du = super().__call__(u)
        if isinstance(dt, ConstantSeries):
            _dt = dt[0]
        else:
            _dt = dt
        return du / self._dt_denominator(_dt)


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
        case _:
            raise NotImplementedError

    if init is None:
        init = FE

    return FiniteDifference(d, init, f'AB{n}')


AB1 = AB(1)
AB2 = AB(2)


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
        case _:
            raise NotImplementedError

    if init is None:
        init = BE

    return FiniteDifference(d, init, f'AM{n}')


AM1 = AM(1)
AM2 = AM(2)


def BDF(
    n: int,
    init: FiniteDifference | None = None,
) -> FiniteDifferenceDerivative:
    """Backward-differentiation-formulae family of `∂.../∂t` discretizations"""
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
        case _:
            raise NotImplementedError

    if init is None:
        init = DT

    return FiniteDifferenceDerivative(lambda dt: dt, d, init, f'BDF{n}')


BDF1 = BDF(1)
BDF2 = BDF(2)


def finite_difference_order(*args: FiniteDifference | tuple[FiniteDifference, ...] | Any) -> int | None:

    args_fd = [a for a in args if isinstance(a, FiniteDifference)]
    for a in args:
        if isinstance(a, tuple) and all(isinstance(i, FiniteDifference) for i in a):
            args_fd.extend(a)

    orders = [a.order for a in args_fd]
    if orders:
        return max(orders)
    else:
        return None
    

def finite_difference_discretize(
    expr: Series | tuple[Callable, tuple] | Expr | Function, 
    D_fdm: FiniteDifference | tuple[FiniteDifference, ...],
    solution: FunctionSeries | None = None,
) -> Expr:
    if not isinstance(expr, Series):
        return expr
    if isinstance(D_fdm, tuple):
        if isinstance(expr, tuple):
            expr_func, expr_args = expr
        else:
            assert isinstance(expr, ExprSeries) and expr.relation
            expr_func, expr_args = expr.relation
        if isinstance(D_fdm, tuple):
            trial = [i is solution for i in expr_args]
            return expr_func(*(D_i[trl](i) for trl, D_i, i in zip(trial, D_fdm, expr_args, strict=False)))
        else:
            return D_fdm(expr_func(*expr_args))
    else:
        return D_fdm(expr)