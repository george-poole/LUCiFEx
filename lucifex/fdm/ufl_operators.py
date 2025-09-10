from typing import Callable, TypeVar, Any

import ufl
from dolfinx.fem import Function, FunctionSpace

from ..utils import MultipleDispatchTypeError
from .series import Series, FunctionSeries, Unsolved, UnsolvedType, ExprSeries


T = TypeVar('T')
R = TypeVar('R')
def unary_operator(ufl_func: Callable[[T], R]):

    def _decorator(
        dummy: Callable[[], None],
    ) -> Callable[[T], R] | Callable[[Series], ExprSeries] | Callable[[UnsolvedType], UnsolvedType]:

        assert dummy() is None

        # functools.update_wrapper(target, ufl_func) TODO __doc__ only?
        def _inner(u):
            if u is Unsolved:
                return Unsolved
            elif isinstance(u, Series):
                NAME = 'name'
                if hasattr(u, NAME):
                    name = f'{ufl_func.__name__}({getattr(u, NAME)})'
                else:
                    name = None
                return ExprSeries([ufl_func(ui) for ui in u], name)
            else:
                return ufl_func(u)
        return _inner
    
    return _decorator


T0 = TypeVar('T0')
T1 = TypeVar('T1')
R = TypeVar('R')
def binary_operator(ufl_func: Callable[[T0, T1], R]):

    def _decorator(
        target: Callable[[], None],
    ) -> Callable[[T0, T1], R] | Callable[[Series, Series], ExprSeries] | Callable[[Series, Any], Series] | Callable[[Any, Series], ExprSeries] | Callable[[UnsolvedType, Any], UnsolvedType] | Callable[[Any, UnsolvedType], UnsolvedType]:

        assert target() is None

        # @functools.wraps(ufl_func) TODO __doc__ only?
        def _inner(u, v):
            if u is Unsolved or v is Unsolved:
                return Unsolved
        
            NAME = 'name'
            if hasattr(u, NAME) and hasattr(v, NAME):
                name = f'{ufl_func.__name__}({getattr(u, NAME)},{getattr(u, NAME)})'
            else:
                name = None

            match isinstance(u, Series), isinstance(v, Series):
                case False, False:
                    return ufl_func(u, v)
                case False, True:
                    return ExprSeries([ufl_func(u, i) for i in v], name)
                case True, False:
                    return ExprSeries([ufl_func(ui, v) for ui in u], name)
                case True, True:
                    return ExprSeries([ufl_func(ui, vi) for ui, vi in zip(u, v)], name)

        return _inner

    return _decorator


@unary_operator(ufl.div)
def div(): pass


@unary_operator(ufl.grad)
def grad(): pass


@unary_operator(ufl.curl)
def curl(): pass


@binary_operator(ufl.inner)
def inner(): pass


@binary_operator(ufl.dot)
def dot(): pass
