from typing import Callable, ParamSpec, TypeVar, ParamSpec, Generic, TypeVar
from typing_extensions import Self
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
A = ParamSpec('A')
class DeferredCondition(Generic[P]):
    def __init__(
        self,
        condition: Callable[P, bool],
    ):        
        self._condition = condition

    @classmethod
    def from_args(
        cls, 
        evaluation: Callable[A, bool],
    ) -> Callable[A, Self]: # FIXME Self[[]]
        return lambda *a, **k: cls(defer(evaluation)(*a, **k))
    
    def evaluate(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        return self._condition(*args, **kwargs)

    
Q = ParamSpec('Q')
P = ParamSpec('P')
B = ParamSpec('B')
A = ParamSpec('A')
class DeferredRoutine(DeferredCondition[P], Generic[Q, P]):
    def __init__(
        self,
        routine: Callable[Q, None],
        condition: Callable[P, bool],
    ):
        super().__init__(condition)
        self._routine = routine

    @classmethod
    def from_args(
        cls, 
        routine: Callable[B, bool],
        condition: Callable[A, bool],
    ) -> Callable[B, Callable[A, Self]]: # FIXME Self[[], []]
        def _(*args, **kwargs):
            r = defer(routine)(*args, **kwargs)
            def _inner(*a, **k):
                c = defer(condition)(*a, *k)
                return cls(r, c)
            return _inner
        return _
    
    def execute(
        self, 
        *args: Q.args, 
        **kwargs: Q.kwargs,
    ) -> Callable[P, None]:
        return  lambda *a, **k: self._routine(*args, **kwargs) if self.evaluate(*a, **k) else None
    

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
    

class Writer(DeferredRoutine[[float], [float]]):
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
        self.execute(t)(t)
    


