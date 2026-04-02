from abc import ABC, abstractmethod
from typing import (
    Iterable, Any, Callable,
    TypeAlias, Generic,
    TypeVar, overload,
)
from typing_extensions import Self, Unpack, TypeVarTuple
from functools import partial

from ufl.core.expr import Expr
from ufl import TrialFunction, replace
from dolfinx.fem import Function, Constant

from ..utils.py_utils import OverloadTypeError, str_indexed, create_kws_filterer
from .series import FunctionSeries, ConstantSeries, ExprSeries, Series


class FiniteDifferenceOperator(ABC):
    def __call__(
        self,
        u: Series | Any,
        *args: Any,
        trial: FunctionSeries | bool = False,
        strict: bool = False,
        **kwargs: Any,
    ) -> Expr:
        if not isinstance(u, Series):
            if strict:
                raise TypeError(f"Expected argument of type {Series}, not {type(u)}.")
            else:
                return u
        
        return create_kws_filterer(self._call)(u, *args, trial=trial, strict=strict, **kwargs)
    
    @abstractmethod
    def _call(
        self,
        u: Series,
        *args: Any,
        trial: FunctionSeries | bool = False,
        strict: bool = False,
        **kwargs: Any,
    ):
        ...
    

T = TypeVar('T')
class FiniteDifference(FiniteDifferenceOperator):
    """
    Finite difference operator for time discretization
    """

    def __init__(
        self,
        coefficients: dict[int, float] | tuple[Iterable[int], Iterable[float]],
        initial: 
            tuple[Iterable[int], Iterable[float]] | dict[int, float] | Self | None
        = None,
        name: str | None = None,
        index: int | None = None,
        use_repr: bool = True,
    ) -> None:

        if not isinstance(coefficients, dict):
            coefficients = dict(zip(*coefficients, strict=True))
        self._coefficients = dict(sorted(coefficients.items()))

        if initial is None:
            if self.order > 1:
                raise ValueError(
                    "Must also provide first-order finite difference for initialization"
                )
            self._initial = None
        elif isinstance(initial, FiniteDifference):
            self._initial = initial
        elif isinstance(initial, (dict, tuple)):
            self._initial = FiniteDifference(initial)
        else:
            raise OverloadTypeError(initial)
        
        if self._initial is not None:
            assert self._initial.order in (0, 1)

        if name is None:
            name = f'{self.__class__.__name__}{id(self)}'
        self._name = name
        self._index = index
        self._use_repr = use_repr

    @property
    def coefficients(self) -> dict[int, float]:
        return self._coefficients
    
    @property
    def implicit_coeff(self) -> float:
        return self.coefficients.get(Series.FUTURE_INDEX, 0.0)

    @property
    def order(self) -> int:
        return Series.FUTURE_INDEX - min(self.coefficients)

    @property
    def initial(self) -> Self | None:
        return self._initial

    @property
    def is_implicit(self) -> bool:
        return max(self.coefficients) == Series.FUTURE_INDEX

    @property
    def is_explicit(self) -> bool:
        return not self.is_implicit
    
    @property
    def name(self) -> str:
        if self._index is not None:
            return f'{self._name}{self._index}'
        else:
            return self._name
        
    def __repr__(self):
        if not self._use_repr:
            return self.name
        kws = dict(
            initial=repr(self.initial),
            name=self.name,
            index=self._index,
        )
        kws_str = ', '.join([f'{k}={v}' for k, v in kws.items()])
        return f'{self.__class__.__name__}({kws_str})'
    
    def __str__(self):
        if self._index is not None:
            return str_indexed(self._name, self._index, 'subscript')
        else:
            return self._name
        
    @overload
    def __call__(
        self,
        u: T,
        **kwargs: Any
    ) -> T:
        """
        `𝒟(u) = u` for any argument `u` not of type `Series`.
        """
        ...
        
    @overload
    def __call__(
        self,
        u: Series,
        *,
        trial: FunctionSeries | bool = False,
        strict: bool = False,
        **kwargs: Any
    ) -> Expr:
        """
        `𝒟(u) = Σⱼ αⱼuⁿ⁺ʲ` for an argument `u` of type `Series`
        given a set of indices and coefficients `{(j, αⱼ)}` with `j ≤ 1`.

        e.g. \\
        `u(x, tⁿ) = uⁿ` of type `FunctionSeries` \\
        `u(tⁿ) = uⁿ` of type `ConstantSeries` 
        """
        ...

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def _call(
        self,
        u: Series,
        *,
        trial: FunctionSeries | bool = False,
    ) -> Expr:
            
        if self.order > u.order:
            raise RuntimeError(
                f"Order of finite difference operator '{self.name}' exceeds order of series '{u.name}'",
            )

        expr = sum((c * u[n] for n, c in self.coefficients.items()))

        if trial is True:
            if not isinstance(u, FunctionSeries):
                raise TypeError(
                    f'Can only deduce the trial argument from an argument of type `FunctionSeries`, not {type(u)}'
                )    
            trial = u

        if trial is not False:
            expr = replace(
                expr, 
                {trial[trial.FUTURE_INDEX]: TrialFunction(trial.function_space)},
            )

        return expr
    
    def __matmul__(self, other: Self) -> 'FiniteDifferenceArgwise':
        if not isinstance(other, FiniteDifference):
            raise NotImplementedError
        return FiniteDifferenceArgwise(self, other)
        
    # def __rmatmul__(self, other: Self) -> 'FiniteDifferenceArgwise':
    #     return other.__matmul__(self)

    def __eq__(self, other: Self) -> bool:
        if not isinstance(other, FiniteDifference):
            raise TypeError(f'Cannot test equality of a `FiniteDifference` with a {type(other)}')
        
        return self.coefficients == other.coefficients
    
    def __hash__(self):
        return hash((*self.coefficients.keys(), *self.coefficients.values()))
        

class FiniteDifferenceDerivative(FiniteDifference):

    def __init__(
        self,
        numerator: FiniteDifference 
        | dict[int, float] | tuple[Iterable[int], Iterable[float]],
        denominator: Callable[[Constant], Constant | Expr] | Callable[[float], float],
        initial: 
            tuple[Iterable[int], Iterable[float]] | dict[int, float] | Self | None
        = None,
        name: str | None = None,
        index: str | None = None,
        use_repr: bool = True,
    ) -> None:
        if isinstance(numerator, FiniteDifference):
            coefficients = numerator.coefficients
        else:
            coefficients = numerator
        super().__init__(coefficients, initial, name, index, use_repr)
        self._denominator = denominator

    @overload
    def __call__(
        self,
        u: Series,
        dt: ConstantSeries | Constant | None = None,
        *,
        trial: FunctionSeries | bool = False,
        strict: bool = False,
        **kwargs: Any
    ) -> Expr:
        """
        `∂u/∂t = 𝒟(u, Δt) = 𝒟(u) / D(Δtⁿ)` given the denominator function `D`
        or just `𝒟(u)` if `dt=None`.
        """
        ...

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def _call(
        self, 
        u: Series,
        dt: ConstantSeries | Constant | None = None,
        *,
        trial: FunctionSeries | bool = False,
    ) -> Expr:
        Du = super()._call(u, trial=trial)
        if dt is None:
            return Du
        
        if isinstance(dt, ConstantSeries):
            _dt = dt[dt.FUTURE_INDEX - 1]
        else:
            _dt = dt
        return Du / self._denominator(_dt)
    
    @property
    def denominator(self) -> Callable[[Constant], Constant | Expr] | Callable[[float], float]:
        return self._denominator


CN = FiniteDifference(
    {
        Series.FUTURE_INDEX: 0.5,
        Series.FUTURE_INDEX - 1: 0.5,
    },
    name='CN',
    use_repr=False,
)
"""Crank-Nicolson"""

FE = FiniteDifference({Series.FUTURE_INDEX - 1: 1.0}, name='FE', use_repr=False)
"""Forward Euler explicit method"""

BE = FiniteDifference({Series.FUTURE_INDEX: 1.0}, name='BE', use_repr=False)
"""Backward Euler implicit method"""

DT = FiniteDifferenceDerivative(
    {
        Series.FUTURE_INDEX: 1.0,
        Series.FUTURE_INDEX - 1: -1.0,
    },
    lambda dt: dt,
    name='DT',
    use_repr=False,
)
"""Forward first time derivative `(uⁿ⁺¹ - uⁿ) / Δtⁿ` """

DTLF = FiniteDifferenceDerivative(
    {
        Series.FUTURE_INDEX: 0.5,
        Series.FUTURE_INDEX - 2: -0.5,
    },
    lambda dt: dt,
    DT,
    name='LF',
    use_repr=False,
)
"""Centred (leapfrog) first time derivative `(uⁿ⁺¹ - uⁿ⁻¹) / 2Δtⁿ` """

DT2 = FiniteDifferenceDerivative(
    {
        Series.FUTURE_INDEX: 1.0,
        Series.FUTURE_INDEX - 1: -2.0,
        Series.FUTURE_INDEX - 2: 1.0,
    },
    lambda dt: dt ** 2,
    DT,
    name='DT2',
    use_repr=False,
)
"""Centred second time derivative"""


def AB(n: int, initial: FiniteDifference | None = None) -> FiniteDifference:
    """
    Adams-Bashforth family of explicit methods. 
    `AB(1)` is forward Euler.
    """
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

    if initial is None:
        initial = FE

    return FiniteDifference(d, initial, 'AB', n, use_repr=False)


AB1 = AB(1)
AB2 = AB(2)
AB3 = AB(3)


def AM(
    n: int, 
    initial: FiniteDifference | None = None,
) -> FiniteDifference:
    """
    Adams-Moulton family of implicit methods.
    `AM(1)` is backward Euler and `AM(2)` is Crank-Nicolson.
    """
    match n:
        case 1:
            d = BE.coefficients
        case 2:
            d = CN.coefficients
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

    if initial is None:
        initial = BE

    return FiniteDifference(d, initial, 'AM', n, use_repr=False)


AM1 = AM(1)
AM2 = AM(2)
AM3 = AM(3)


def BDF(
    n: int,
    initial: FiniteDifference | None = None,
) -> FiniteDifferenceDerivative:
    """
    Backward-differentiation-formulae family for discretizations of `∂u/∂t`.
    `BDF(1)` is equivalent to `DT`.
    """
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

    if initial is None:
        initial = DT

    return FiniteDifferenceDerivative(d, lambda dt: dt, initial, 'BDF', n, use_repr=False)


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


ExplicitFiniteDifference: TypeAlias = FiniteDifference
"""
Type alias to any `FiniteDifference` that is explicit.
"""

ImplicitFiniteDifference: TypeAlias = FiniteDifference
"""
Type alias to any `FiniteDifference` that is implicit.
"""    


T = TypeVar('T')
FDS = TypeVarTuple('FDS')  
class FiniteDifferenceArgwise(
    Generic[Unpack[FDS]],
    FiniteDifferenceOperator,
):
    """
    Argument-wise application of finite difference operators to time-dependent
    arguments of an expression.

    To combine individual `FiniteDifference` operators into a `FiniteDifferenceArgwise` operator, 
    use either the class constructor or the `@` infix.

    e.g. `FiniteDifferenceArgwise(AB2, CN)` or `AB2 @ CN`
    """
    def __init__(
        self, 
        *fd_args: FiniteDifference,
        name: str | None = None,
        use_repr: bool = False,
    ):
        self._fd_args = tuple(fd_args)
        self._name = name
        self._use_repr = use_repr

    @overload
    def __call__(
        self,
        u: T,
        **kwargs: Any
    ) -> T:
        """
        `𝒟(u) = u` for any argument `u` not of type `Series`.
        """
        ...
        
    @overload
    def __call__(
        self,
        u: Series,
        *,
        trial: FunctionSeries | bool = False,
        strict: bool = False,
        args: Iterable[Any] | None = None,
        fd_template: FiniteDifference = FE,
    ) -> Expr:
        """
        `(𝒟₀◦𝒟₁◦...)(f(u₀, u₁, ...)) = f(𝒟₀(u₀), 𝒟₁(u₁), ...)`
        """
        ...

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
    def _call(
        self, 
        u: ExprSeries | Any, 
        *,
        trial: FunctionSeries | bool = False,
        strict: bool = False,
        args: Iterable[Any] | None = None,
        fd_replaced: FiniteDifference = FE,
    ) -> Expr | Any:
        """
        `(𝒟₀◦𝒟₁◦...)(f(u₀, u₁, ...)) = f(𝒟₀(u₀), 𝒟₁(u₁), ...)`
        """            
        if strict and args is not None:
            for i in args:
                if not i in u.args:
                    raise RuntimeError(
                        f'{i} is not an argument to the `ExprSeries`.'
                    )
            
        if args is None:
            args = u.args

        finite_differences = [partial(fd, trial=trial) for fd in self._fd_args]

        mapping = {
            fd_replaced(a): fd(a) 
            for a, fd in zip(args, finite_differences, strict=True)
        }
        return replace(fd_replaced(u), mapping)
        # return u.reconstruct_expr(*finite_differences, args=args)

    @property
    def finite_differences(self) -> tuple[FiniteDifference, ...]:
        return self._fd_args
    
    @property
    def order(self) -> int:
        return max(i.order for i in self.finite_differences)
    
    @property
    def initial(self) -> Self | None:
        if self.order <= 1:
            return None
        fd_args_init = [fd.initial if fd.initial is not None else fd for fd in self._fd_args]
        return FiniteDifferenceArgwise(*fd_args_init)

    def __iter__(self):
        return iter(self.finite_differences)
    
    @property
    def name(self) -> str:
        if self._name is None:
            return ' @ '.join([repr(fd) for fd in self.finite_differences])
        else:
            return self._name
    
    def __repr__(self):
        if not self._use_repr:
            return self.name
        return f'{self.__class__.__name__}({self._fd_args}, {self.name})'
    
    def __str__(self):
        if self._name is None:
            return '◦'.join([str(fd) for fd in self.finite_differences])
        else:
            return self._name
    
    def __matmul__(
        self, 
        other: Self | FiniteDifference,
    ) -> 'FiniteDifferenceArgwise':
        if isinstance(other, FiniteDifference):
            return FiniteDifferenceArgwise(*self._fd_args, other)
        if isinstance(other, FiniteDifferenceArgwise):
            return FiniteDifferenceArgwise(
                *self._fd_args,
                *other._fd_args, 
            )
        raise NotImplementedError
        
    def __rmatmul__(
        self, 
        other: Self | FiniteDifference,
    ) -> 'FiniteDifferenceArgwise':
        if isinstance(other, FiniteDifference):
            return FiniteDifferenceArgwise(other, *self._fd_args, other)
        if isinstance(other, FiniteDifferenceArgwise):
            return other.__matmul__(self)
        raise NotImplementedError
    
    def __eq__(self, other: Self | FiniteDifference) -> bool:
        if isinstance(other, FiniteDifference):
            return False
        if not isinstance(other, FiniteDifferenceArgwise):
            raise TypeError(f'Cannot test equality of a `FiniteDifference` with a {type(other)}')
        
        if len(self.finite_differences) != len(other.finite_differences):
            return False
        
        return all(
            i == j 
            for i, j in zip(self.finite_differences, other.finite_differences, strict=True)
        )
    
    def __hash__(self):
        arg = []
        for fd in self.finite_differences:
            arg.append((*fd.coefficients.keys(), *fd.coefficients.values()))
        return hash(tuple(arg))
    

def finite_difference_order(
    *args: FiniteDifference | FiniteDifferenceArgwise | Any,
    minimum: int = 1,
    strict: bool = False,
) -> int | None:
    orders = [a.order for a in args if isinstance(a, (FiniteDifference, FiniteDifferenceArgwise))]
    if strict and len(orders) < len(args):
        raise TypeError('All arguments must be of type `FiniteDifference` or `FiniteDifferenceArgwise`.')
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
        super().__init__(f"{required} discretization required, not {fd}. {msg}")


class ExplicitDiscretizationError(DiscretizationError):
    def __init__(
        self, 
        fd: FiniteDifference,
        msg: str = '',
    ):
        super().__init__('Implicit', fd, msg)


class ImplicitDiscretizationError(DiscretizationError):
    def __init__(
        self, 
        fd: FiniteDifference,
        msg: str = '',
    ):
        super().__init__('Explicit', fd, msg)