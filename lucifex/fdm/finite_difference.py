from typing import Iterable, Any, Callable, overload
from typing_extensions import Self

from ufl.core.expr import Expr
from ufl import TrialFunction
from dolfinx.fem import Function, Constant

from ..utils.str_utils import str_indexed 
from .series import FunctionSeries, ConstantSeries, ExprSeries, Series


class FiniteDifference:
    """
    Finite difference operator for a time-dependent argument
    """

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
            name = f'{self.__class__.__name__}{id(self)}'
        self._name = name

        self._trial = self.trial_default
    
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
    def init(self) -> Self | None:
        return self._init

    @property
    def is_implicit(self) -> bool:
        return max(self.coefficients) == Series.FUTURE_INDEX

    @property
    def is_explicit(self) -> bool:
        return not self.is_implicit
    
    def __str__(self):
        return self.name

    @overload
    def __call__(
        self,
        u: Series,
        trial: bool | None = None,
        strict: bool = False,
    ) -> Expr:
        ...

    @overload
    def __call__(
        self,
        u: Expr | Function | Constant,
        **kwargs,
    ) -> Expr | Function | Constant:
        ...

    def __call__(
        self,
        u: Series | Expr | Function | Constant,
        trial: bool | FunctionSeries | None = None,
        strict: bool = False,
    ) -> Expr:
        """
        `𝒟(u(x, tⁿ)) = Σⱼ αⱼu(x)ⁿ⁺ʲ` given a set of indices and 
        coefficients `{(j, αⱼ)}` with `j ≤ 1`.
        """
        if not isinstance(u, Series):
            if strict:
                raise TypeError(f"Expected argument of type {Series}, not {type(u)}.")
            else:
                return u
            
        if self.order > u.order:
            raise RuntimeError(
                f"Order of finite difference operator '{self.name}' exceeds order of series '{u.name}'",
            )

        if isinstance(trial, FunctionSeries):
            trial = u is trial

        if trial is None:
            trial = self._trial if isinstance(u, FunctionSeries) else False

        assert isinstance(trial, bool)
        if trial:
            if not isinstance(u, FunctionSeries):
                raise TypeError(f'Expected `FunctionSeries` type, not `{type(u)}`.')
            _u = lambda n: u[n] if n != u.FUTURE_INDEX else TrialFunction(u.function_space)
        else:
            _u = lambda n: u[n]

        return sum((c * _u(n) for n, c in self.coefficients.items()))
    
    def __matmul__(self, other: Self) -> 'FiniteDifferenceArgwise':
        if not isinstance(other, FiniteDifference):
            raise NotImplementedError
        return FiniteDifferenceArgwise(self, other)
        
    # def __rmatmul__(self, other: Self) -> 'FiniteDifferenceArgwise':
    #     return other.__matmul__(self)
        
        

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

    return FiniteDifference(d, init, str_indexed('AB', n, 'subscript'))


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

    return FiniteDifference(d, init, str_indexed('AM', n, 'subscript'))


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

    return FiniteDifferenceDerivative(lambda dt: dt, d, init, str_indexed('BDF', n, 'subscript'))


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


class FiniteDifferenceArgwise:
    """
    Argument-wise application of finite difference operators to time-dependent
    arguments of an expression.

    To combine individual `FiniteDifference` operators into a `FiniteDifferenceArgwise` operator, 
    use either the class constructor or the `@` infix.

    e.g. `FiniteDifferenceArgwise(AB2, CN)` or `AB2 @ CN`
    """
    def __init__(
        self, 
        *args: FiniteDifference,
        name: str | None = None,
        **kws: FiniteDifference,
    ):
        self._fd_args = tuple(args)
        self._fd_kws = kws.copy()
        if name is None:
            name = '◦'.join([fd.name for fd in self.finite_differences])
        self._name = name

    @overload
    def __call__(
        self, 
        u: ExprSeries, 
        /,
        trial: FunctionSeries | None = None,
        strict: bool = False,
    ) -> Expr:
        ...

    @overload
    def __call__(
        self, 
        *args: Any | Callable[..., Expr], 
        trial: FunctionSeries | None = None,
        strict: bool = False,
    ) -> Expr:
        ...

    @overload
    def __call__(
        self,
        u: Expr | Function | Constant,
        /,
        **kwargs,
    ) -> Expr | Function | Constant:
        ...
    
    def __call__(
        self, 
        *args,
        trial=None,
        strict=False,
    ):
        """
        `(𝒟₀◦𝒟₁◦...)(f(u₀, u₁, ...)) = f(𝒟₀(u₀), 𝒟₁(u₁), ...)`
        """
        if not args:
            raise TypeError
        
        if len(args) > 1:
            *args, func = args
            kws = {}
        else:
            u = args[0]
            if not isinstance(u, ExprSeries):
                if strict:
                    raise TypeError(f"Expected argument of type {ExprSeries} or {tuple}, not {type(u)}.")
                else:
                    return u
            if not u.expression:
                raise ValueError(f"Expression and arguments must be deducable '{u.name}'.")
            func, args, kws = u.expression

        _args = [fd(arg, arg is trial) for fd, arg in zip(self._fd_args, args, strict=False)]
        _kws = {k: fd(kws[k], kws[k] is trial) for k, fd in self._fd_kws.items()}
        
        return func(*_args, **_kws)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def finite_differences(self) -> tuple[FiniteDifference, ...]:
        return tuple((*self._fd_args, *self._fd_kws.values()))
    
    @property
    def order(self) -> int:
        return max(i.order for i in self.finite_differences)
    
    @property
    def init(self) -> Self | None:
        if self.order <= 1:
            return None
        fd_args_init = [fd.init if fd.init is not None else fd for fd in self._fd_args]
        fs_kws_init = {k: v.init if v.init is not None else v for k, v in self._fd_kws.items()}
        return FiniteDifferenceArgwise(*fd_args_init, **fs_kws_init)

    def __iter__(self):
        return iter(self.finite_differences)
    
    def __str__(self):
        return self.name
    
    def __matmul__(
        self, 
        other: Self | FiniteDifference,
    ) -> 'FiniteDifferenceArgwise':
        if isinstance(other, FiniteDifference):
            return FiniteDifferenceArgwise(*self._fd_args, other, **self._fd_kws)
        if isinstance(other, FiniteDifferenceArgwise):
            return FiniteDifferenceArgwise(
                *self._fd_args,
                *other._fd_args, 
                **self._fd_kws,
                **other._fd_kws,
            )
        raise NotImplementedError
        
    def __rmatmul__(
        self, 
        other: Self | FiniteDifference,
    ) -> 'FiniteDifferenceArgwise':
        if isinstance(other, FiniteDifference):
            return FiniteDifferenceArgwise(other, *self._fd_args, other, **self._fd_kws)
        if isinstance(other, FiniteDifferenceArgwise):
            return other.__matmul__(self)
        raise NotImplementedError
    

def finite_difference_order(
    *args: FiniteDifference | FiniteDifferenceArgwise | Any,
    minimum: int = 1,
) -> int | None:
    orders = [a.order for a in args if isinstance(a, (FiniteDifference, FiniteDifferenceArgwise))]
    if orders:
        return max(minimum, max(orders))
    else:
        return None
        

class DiscretizationError(RuntimeError):
    def __init__(
        self, 
        required: str,
        fd: FiniteDifference,
        msg: str = '',
    ):
        super().__init__(f"{required} discretization required, not {fd}'. {msg}")


class ImplicitDiscretizationError(DiscretizationError):
    def __init__(
        self, 
        fd: FiniteDifference,
        msg: str = '',
    ):
        super().__init__('Implicit', fd, msg)


class ExplicitDiscretizationError(DiscretizationError):
    def __init__(
        self, 
        fd: FiniteDifference,
        msg: str = '',
    ):
        super().__init__('Explicit', fd, msg)

