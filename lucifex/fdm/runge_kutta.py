from typing import (
    Iterable, ParamSpec, Callable, overload,
    Protocol, Concatenate, TypeAlias,
)

import numpy as np
from ufl.core.expr import Expr
from dolfinx.fem import Function, Constant

from .series import ConstantSeries, FunctionSeries
from .finite_difference import FE


class RungeKuttaRHS(Protocol):

    @overload
    def __call__(
        self, 
        t: Constant | ConstantSeries,
        u: Function | FunctionSeries,
    ) -> Expr:
        ...

    @overload
    def __call__(
        self, 
        u: Function | FunctionSeries,
    ) -> Expr:
        ...


P = ParamSpec('P')
class RungeKutta:
    """
    Discretization of `∂u/∂t = f(t, y)` to

    `(uⁿ⁺¹ - uⁿ) / Δtⁿ = ∑ᵢ bᵢkᵢ`

    where

    `kᵢ = f(tⁿ + cᵢΔtⁿ, uⁿ + Δtⁿ∑ⱼaᵢⱼkⱼ)`.

    Explicit if `aᵢⱼ = 0` for all `i ≤ j`.
    """
    def __init__(
        self,
        a_dense: Iterable[Iterable[float]],
        b: Iterable[float],
        c: Iterable[float],
        name: str | None = None,
    ):
        n_stages = len(b)
        assert n_stages == len(c)

        a_dense = np.array(a_dense)
        assert a_dense.shape == (n_stages, n_stages)
        b = np.array(b)
        c = np.array(c)
        
        self._a = a_dense
        self._b = b
        self._c = c
        self._name = name

    @property
    def n_stages(self) -> int:
        return len(self._b)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b
    
    @property
    def c(self):
        return self._c
    
    @property
    def butcher(self) -> str:
        _butcher = ''
        for i in range(self.n_stages):
            _butcher = f'{_butcher}\n{self.c[i]} |'
            for a_ij in self.a[i, :]:
                _butcher = f'{_butcher} {a_ij}'

        underscores = '__' * (self.n_stages + 1)
        _butcher = f'{_butcher}\n {underscores}'
        _butcher = f'{_butcher}\n  |'
        for bi in self.f:
            _butcher = f'{_butcher} {bi}'

        return _butcher

    @property
    def name(self) -> str | None:
        return self._name


T: TypeAlias = Constant | ConstantSeries
U: TypeAlias = Function | FunctionSeries
V: TypeAlias = Function | FunctionSeries
P = ParamSpec('P')
class ExplicitRungeKutta(RungeKutta):
    def __init__(
        self,
        a_triangular: Iterable[Iterable[float]],
        b: Iterable[float],
        c: Iterable[float],
        name: str | None = None,
    ):
        n_stages = len(b)
        a_dense = np.zeros((n_stages, n_stages))
        for i in range(1, n_stages):
            a_dense[i, :] = a_triangular[i]

        if len(c) == len(b) - 1:
            c = (0, *c)

        super().__init__(a_dense, b, c, name)

    def __call__(
        self, 
        rhs: Callable[Concatenate[T, U, P], Expr],
        dt,
        autonomous: bool,
    ):    
        
        if isinstance(dt, ConstantSeries):
                dt = dt[0]


        def _(
            t: Constant | ConstantSeries,
            u: FunctionSeries,
            *args: P.args, 
            **kwargs: P.kwargs,
        ) -> Expr:
            if isinstance(t, ConstantSeries):
                t = t[0]

            k = [None] * len(self.n_stages)
            for i in range(self.n_stages):
                a_row = self.a[i, :]
                a_row_sum = sum(a * _k for a, _k in zip(a_row[:i], k))
                k[i] = rhs(
                    t + self.c[i] * dt, 
                    u[0] + dt * a_row_sum,
                    *[FE(i) for i in args],
                    **{n: FE(v) for n, v in kwargs.items()}
                )
            assert k.count(None) == 0

            return sum(b * _k for b, _k in zip(self.b, k))

        return _


RK4 = None 
# ExplicitRungeKutta(
#     [(0.5, ), (0.0, 0.5), (0.0, 0.0, 1.0)],
#     (1/6, 1/3, 1/3, 1/6),
#     (0.5, 0.5, 1.0),
#     'RK4',
# )
# """
# Explicit RK4 method
# """
MID = None
# MID = ExplicitRungeKutta(
#     [(0.5, )],
#     (0.0, 1.0),
#     (0.5, ),
#     'MID'
# )
# """
# Explicit midpoint method
# """