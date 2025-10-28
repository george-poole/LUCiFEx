from typing import Callable, ParamSpec, TypeVar, ParamSpec, Generic, TypeVar
from math import floor
from inspect import signature

import numpy as np
from dolfinx.fem import Constant

P = ParamSpec("P")
R = TypeVar('R')
def defer(func: Callable[P, R]) -> Callable[P, Callable[[], R]]:
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
class DeferredCondition(DeferredEvaluation[P, bool], Generic[P]):
    pass


P = ParamSpec('P')
class DeferredRoutine(DeferredEvaluation[P, None], Generic[P]):
    pass


P = ParamSpec('P')
Q = ParamSpec('Q')
R = TypeVar('R')
class DeferredConditionalEvaluation(DeferredEvaluation[P, R], Generic[P, Q, R]):
    def __init__(
        self,
        func: Callable[P, R],
        condition: Callable[Q, bool],
    ):
        super().__init__(func)
        self._condition = DeferredCondition(condition)

    def evaluate_conditional(
        self, 
        *args: P.args, 
        **kwargs: P.kwargs,
    ) -> Callable[Q, R | None]:
        return lambda *a, **k: self._func(*args, **kwargs) if self._condition.evaluate(*a, **k) else None
    
    def conditional_evaluate(
        self, 
        *args: Q.args, 
        **kwargs: Q.kwargs,
    ) -> Callable[P, R | None]:
        return lambda *a, **k: self._func(*a, **k) if self._condition.evaluate(*args, **kwargs) else None
    

P = ParamSpec('P')
Q = ParamSpec('Q')
class DeferredConditionalRoutine(DeferredConditionalEvaluation[P, Q, None], Generic[P, Q]):
    pass
    

class Stopper(DeferredCondition[[float]]):
    def __init__(
        self, 
        condition: Callable[[], bool] | Callable[[float], bool] | int | float,
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

        super().__init__(inject_time_arg(condition))

    def stop(self, t: float | Constant | np.ndarray) -> bool:
        return self.evaluate(float(t))
    

class Writer(DeferredConditionalRoutine[[float], [float]]):
    def __init__(
        self,
        routine: Callable[[], None] | Callable[[float], None],
        condition: Callable[[], bool] | Callable[[float], bool] | int | float | None = None,
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

        super().__init__(inject_time_arg(routine), inject_time_arg(condition))
        if name is None:
            name = f'{self.__class__.__name__}{id(self)}'
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def write(self, t: float | Constant | np.ndarray) -> None:
        t = float(t)
        self.evaluate_conditional(t)(t)


def inject_time_arg(
    condition: Callable[[], bool] | Callable[[float], bool],
) -> Callable[[float], bool]:
    n_args = len(signature(condition).parameters)
    if n_args == 0:
        return lambda _: condition()
    else:
        if not n_args == 1:
            raise TypeError('Expected callable with at most one argument.')
        return lambda t: condition(t)
    


# P = ParamSpec('P')
# class DeferredCondition(Generic[P]):
#     def __init__(
#         self,
#         condition: Callable[P, bool],
#     ):        
#         self._condition = condition
    
#     def evaluate(self, *args: P.args, **kwargs: P.kwargs) -> bool:
#         return self._condition(*args, **kwargs)

    
# Q = ParamSpec('Q')
# P = ParamSpec('P')
# class DeferredRoutine(DeferredCondition[P], Generic[Q, P]):
#     def __init__(
#         self,
#         routine: Callable[Q, None],
#         condition: Callable[P, bool],
#     ):
#         super().__init__(condition)
#         self._routine = routine

#     def execute(
#         self, 
#         *args: Q.args, 
#         **kwargs: Q.kwargs,
#     ) -> Callable[P, None]:
#         return  lambda *a, **k: self._routine(*args, **kwargs) if self.evaluate(*a, **k) else None
    


