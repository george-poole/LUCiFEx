from typing import (
    Callable, ParamSpec, TypeVar, 
    ParamSpec, Generic, TypeVar, TypeAlias,
)
from math import floor

import numpy as np
from dolfinx.fem import Constant

from .func_utils import arity


T = TypeVar('T')
LazyEvaluator: TypeAlias = Callable[[], T]


P = ParamSpec("P")
R = TypeVar('R')
def create_lazy_evaluator(func: Callable[P, R]) -> Callable[P, LazyEvaluator[R]]:
    def _(*args: P.args, **kwargs: P.kwargs):
        return lambda: func(*args, **kwargs)
    return _


P = ParamSpec('P')
R = TypeVar('R')
class DeferredEvaluation(Generic[P, R]):
    def __init__(
        self,
        func: Callable[P, R],
    ):        
        self._func = func
    
    def evaluate(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)
    

P = ParamSpec('P')
class DeferredBoolean(DeferredEvaluation[P, bool], Generic[P]):
    pass


P = ParamSpec('P')
class DeferredRoutine(DeferredEvaluation[P, None], Generic[P]):
    pass


P = ParamSpec('P')
Q = ParamSpec('Q')
R = TypeVar('R')
class DeferredConditional(DeferredEvaluation[P, R], Generic[P, Q, R]):
    def __init__(
        self,
        func: Callable[P, R],
        condition: Callable[Q, bool],
    ):
        super().__init__(func)
        self._condition = DeferredBoolean(condition)

    def evaluate_if_true(
        self, 
        *args: P.args, 
        **kwargs: P.kwargs,
    ) -> Callable[Q, R | None]:
        return lambda *a, **k: self._func(*args, **kwargs) if self._condition.evaluate(*a, **k) else None
    

P = ParamSpec('P')
Q = ParamSpec('Q')
class DeferredConditionalRoutine(DeferredConditional[P, Q, None], Generic[P, Q]):
    pass
    

class Stopper(DeferredBoolean[[float]]):
    def __init__(
        self, 
        condition: LazyEvaluator[bool] | Callable[[float], bool] | int | float,
    ):
        if isinstance(condition, int):
            n_stop = condition
            n_step = 0
            def condition() -> bool:
                nonlocal n_step
                if n_step >= n_stop:
                    r = True
                else:
                    r = False
                n_step += 1
                return r

        if isinstance(condition, float):
            t_stop = condition
            condition = lambda t: t > t_stop

        super().__init__(inject_float_arg(condition))

    def stop(self, t: float | Constant | np.ndarray) -> bool:
        return self.evaluate(float(t))
    

class Writer(DeferredConditionalRoutine[[float], [float]]):
    def __init__(
        self,
        routine: LazyEvaluator[None] | Callable[[float], None],
        condition: LazyEvaluator[bool] | Callable[[float], bool] | int | float | None = None,
        name: str | None = None,
    ):
        if condition is None:
            condition = lambda: True

        if isinstance(condition, int):
            n_stop = condition
            n_step = 0
            def condition() -> bool:
                nonlocal n_step
                if n_step % n_stop == 0:
                    r = True
                else:
                    r = False
                n_step += 1
                return r

        if isinstance(condition, float):
            dt = condition
            n_dt = 0
            def condition(t: float) -> bool:
                nonlocal n_dt
                _n = floor(t / dt)
                if _n >= n_dt:
                    n_dt = max(n_dt + 1, _n)
                    return True
                else:
                    return False  

        super().__init__(inject_float_arg(routine), inject_float_arg(condition))
        if name is None:
            name = f'{self.__class__.__name__}{id(self)}'
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def write(self, t: float | Constant | np.ndarray) -> None:
        t = float(t)
        self.evaluate_if_true(t)(t)


def inject_float_arg(
    condition: LazyEvaluator[bool] | Callable[[float], bool],
) -> Callable[[float], bool]:
    n_args = arity(condition)
    if n_args == 0:
        return lambda _: condition()
    else:
        if not n_args == 1:
            raise TypeError('Expected callable with at most one argument.')
        return lambda t: condition(t)



