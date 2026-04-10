from typing import Callable, TypeVar, Any, ParamSpec, Concatenate

import ufl

from ..utils.py_utils import replicate_callable
from .series import Series, ExprSeries, UnsolvedType, Unsolved, extract_args_from_series


T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')
def unary_overload(
    ufl_func: Callable[Concatenate[T, P], R],
) -> (
    Callable[Concatenate[T, P], R] 
    | Callable[Concatenate[Series, P], ExprSeries] 
    | Callable[Concatenate[UnsolvedType, P], UnsolvedType]
):
    NAME = 'name'
    # functools.update_wrapper(dummy, ufl_func) #TODO __doc__ only?
    def _(u, *a, **k):
        if u is Unsolved:
            return Unsolved
        elif isinstance(u, Series):
            if hasattr(u, NAME):
                name = f'{ufl_func.__name__}({getattr(u, NAME)})'
            else:
                name = None
            return ExprSeries([ufl_func(ui) for ui in u], name, args=extract_args_from_series(u))
        else:
            return ufl_func(u, *a, **k)
        
    return _


T0 = TypeVar('T0')
T1 = TypeVar('T1')
P = ParamSpec('P')
R = TypeVar('R')
def binary_overload(
    ufl_func: Callable[Concatenate[T0, T1, P], R],
) -> (
    Callable[Concatenate[T0, T1, P], R] 
    | Callable[Concatenate[Series, Series, P], ExprSeries] 
    | Callable[Concatenate[Series, Any, P], ExprSeries] | Callable[Concatenate[Any, Series, P], ExprSeries] 
    | Callable[Concatenate[UnsolvedType, Any, P], UnsolvedType] | Callable[Concatenate[Any, UnsolvedType, P], UnsolvedType]
):
    NAME = 'name'
    # @functools.wraps(ufl_func) TODO __doc__ only?
    def _(u, v, *a, **k):
        if u is Unsolved or v is Unsolved:
            return Unsolved
    
        if hasattr(u, NAME) and hasattr(v, NAME):
            name = f'{ufl_func.__name__}({getattr(u, NAME)},{getattr(u, NAME)})'
        else:
            name = None

        args = extract_args_from_series(u, v)

        match isinstance(u, Series), isinstance(v, Series):
            case False, False:
                return ufl_func(u, v)
            case False, True:
                return ExprSeries([ufl_func(u, i, *a, **k) for i in v], name, args=args)
            case True, False:
                return ExprSeries([ufl_func(ui, v, *a, **k) for ui in u], name, args=args)
            case True, True:
                return ExprSeries([ufl_func(ui, vi, *a, **k) for ui, vi in zip(u, v)], name, args=args)

    return _


@replicate_callable(unary_overload(ufl.div))
def div(): pass


@replicate_callable(unary_overload(ufl.grad))
def grad(): pass


@replicate_callable(unary_overload(ufl.curl))
def curl(): pass


@replicate_callable(unary_overload(ufl.nabla_div))
def nabla_div(): pass


@replicate_callable(unary_overload(ufl.nabla_grad))
def nabla_grad(): pass


@replicate_callable(unary_overload(ufl.Dx))
def Dx(): pass


@replicate_callable(unary_overload(ufl.exp))
def exp(): pass


@replicate_callable(unary_overload(ufl.sin))
def sin(): pass


@replicate_callable(unary_overload(ufl.cos))
def cos(): pass


@replicate_callable(unary_overload(ufl.tan))
def tan(): pass


@replicate_callable(unary_overload(ufl.asin))
def asin(): pass


@replicate_callable(unary_overload(ufl.acos))
def acos(): pass


@replicate_callable(unary_overload(ufl.atan))
def atan(): pass


@replicate_callable(unary_overload(ufl.atan_2))
def atan_2(): pass


@replicate_callable(unary_overload(ufl.sinh))
def sinh(): pass


@replicate_callable(unary_overload(ufl.cosh))
def cosh(): pass


@replicate_callable(unary_overload(ufl.tanh))
def tanh(): pass


@replicate_callable(binary_overload(ufl.inner))
def inner(): pass


@replicate_callable(binary_overload(ufl.dot))
def dot(): pass


@replicate_callable(binary_overload(ufl.max_value))
def max_value(): pass


@replicate_callable(binary_overload(ufl.min_value))
def min_value(): pass


# TODO as_vector and as_matrix