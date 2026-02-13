from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Iterable, Generic, Protocol, ParamSpec, overload
from typing_extensions import Self
from functools import singledispatch

import numpy as np
from ufl import split
from ufl.core.expr import Expr
from dolfinx.fem import FunctionSpace, Expression
from dolfinx.mesh import Mesh

from ..utils.fenicsx_utils import set_constant, set_function, extract_mesh, create_function_space
from ..utils.py_utils import MultipleDispatchTypeError, Writer
from ..fem.perturbation import Perturbation
from ..fem import Function, Constant, Unsolved, UnsolvedType, is_unsolved


T = TypeVar('T')
class Series(ABC, Generic[T]):
    """
    Abstract base class for a series representing a time-dependent quantity.
    """
    FUTURE_INDEX = 1

    @abstractmethod
    def __init__(
        self,
        create_container: Callable[[int], T],
        name: str | tuple[str, Iterable[str]] | None,
        order: int,
    ):
        assert order >= 1
        self._future = create_container(self.FUTURE_INDEX)
        self._present = create_container(self.FUTURE_INDEX - 1)
        self._previous = [create_container(i) for i in range(-1, -order, -1)][::-1]

        if isinstance(name, tuple):
            name, subnames = name
            subnames = tuple(subnames)
        else:
            subnames = None

        if name is None:
            name = self.__class__.__name__

        self.name = name
        self._subnames = subnames
        self._create_subname = lambda i: (
            self._subnames[i] if self._subnames else f'{self.name}{i}'
        )
    
    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """Time-independent mesh that the series is defined on."""
        ... 

    @property
    @abstractmethod
    def series(self) -> list[T]:
        """
        `[u⁰, u¹, u², ...]`
        """
        ...

    @property
    @abstractmethod
    def time_series(self) -> list[float | None]:
        """
        `[t⁰, t¹, t², ...]`
        """
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError(f'{value} is not a string.')
        self._name = value

    @property
    def order(self) -> int:
        return len(self._previous) + 1

    @property
    def sequence(
        self,
    ) -> tuple[T, ...]:
        """
        `(..., uⁿ⁻¹, uⁿ, uⁿ⁺¹)`
        """
        return tuple((*self._previous, self._present, self._future))
    
    def __getitem__(
        self,
        index: int,
    ) -> T:
        index_min = self.FUTURE_INDEX - self.order
        index_max = self.FUTURE_INDEX

        if index > index_max or index < index_min:
            raise IndexError(
                f"Time index {index} is outside the interval [{index_min}, {index_max}] permitted by order {self.order}"
            )
        elif index == self.FUTURE_INDEX:
            f = self._future
        elif index == self.FUTURE_INDEX - 1:
            f = self._present
        else:
            f = self._previous[index + 1 - self.FUTURE_INDEX]

        return f

    def __setitem__(self, _):
        raise RuntimeError("Not permitted")

    def __str__(self) -> str:
        seq = [i if not is_unsolved(i) else Unsolved for i in self.sequence]
        seq_str = [str(s) for s in seq]
        previous = ', '.join(seq_str[:-2])
        present, future = seq_str[-2:]
        if previous:
            seq_repr = f"{previous}, {present}, {future}"
        else:
            seq_repr = f"{present}, {future}"
        return f"[{seq_repr}]"

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self) -> int:
        return len(self.sequence)

    def __reversed__(self):
        return self.sequence[::-1]

    def __add__(self, other: Any):
        return self.__arithmetic(other, "__add__")
    
    def __radd__(self, other: Any):
        return self.__arithmetic(other, "__add__")

    def __sub__(self, other: Any):
        return self.__arithmetic(other, "__sub__")
    
    def __rsub__(self, other: Any):
        return -1 * (self - other)

    def __mul__(self, other: Any):
        return self.__arithmetic(other, "__mul__")

    def __rmul__(self, other: Any):
        return self.__arithmetic(other, "__mul__")

    def __truediv__(self, other: Any):
        return self.__arithmetic(other, "__truediv__")
    
    def __rtruediv__(self, other: Any):
        return (self / other) ** -1
    
    def __pow__(self, other: Any):
        return self.__arithmetic(other, "__pow__")
    
    def __neg__(self):
        return -1 * self

    def __arithmetic(self, other: Any, method: str):
        if isinstance(other, Series):
            seq_reversed = [
                getattr(i, method)(j) for i, j in zip(reversed(self), reversed(other))
            ]
            seq = seq_reversed[::-1]
            time_series = lambda: [i for i, j in zip(self.time_series, other.time_series) if np.isclose(i, j)]
            series = lambda: [getattr(i, method)(j) for i, j, t in zip(self.series, other.series, self.time_series) if t in time_series()]
        else:
            seq = [getattr(i, method)(other) for i in self]
            time_series = lambda: self.time_series
            series = lambda: [getattr(i, method)(other) for i in self.series]

        return ExprSeries(seq, series=series, time_series=time_series)


P = ParamSpec('P')
class ExprSeries(
    Series[Expr],
):
    _func_args = None

    @overload
    def __init__(
        self,
        arg: Self,
        name: str | None = None,
    ):
        ...

    @overload
    def __init__(
        self,
        arg: Iterable[Expr],
        name: str | None = None,
        series: Callable[[], list[Expr]] | None = None,
        time_series: Callable[[], list[float]] | None = None,
    ):
        ...

    def __init__(
        self,
        arg,
        name: str | None = None,
        series: Callable |  None = None,
        time_series: Callable |  None = None,
    ):
        if isinstance(arg, ExprSeries):
            if name is None:
                name = arg.name
            self.__init__(arg.sequence, name, arg._series, arg._time_series)
        elif isinstance(arg, Series):
            self.__init__(1.0 * arg, name)
        else:
            order = len(arg) - 1
            super().__init__(lambda i: arg[i - self.FUTURE_INDEX - 1], name, order)
            self._series = series
            self._time_series = time_series

    @property
    def series(self):
        if self._series is None:
            return []
        return self._series()
    
    @property
    def time_series(self):
        if self._time_series is None:
            return []
        return self._time_series()

    @overload
    @classmethod
    def _from_expr(
        cls, 
        func: Callable[P, Self], 
        /,
        *,
        name: str | None = None,
    ) -> Callable[P, Self]:
        ...

    @overload
    @classmethod
    def _from_expr(
        cls, 
        *args: Any | Callable[..., Self],
        name: str | None = None,
    ) -> Self:
        ...

    @classmethod
    def _from_expr(
        cls, 
        *args,
        name: str | None = None,
    ):
        if not args:
            raise TypeError
        
        def _(f, *a, **k):
            expr = f(*a, **k)
            obj = cls(expr, name)
            obj._func_args = (f, a, k)
            return obj
        
        if len(args) > 1:
            *args, func = args
            return _(func, *args)
        else:
            func = args[0]
            return lambda *a, **k: _(func, *a, **k)
        
    @classmethod
    def from_expr_func(
        cls, 
        func: Callable[P, Self], 
        /,
        *,
        name: str | None = None,
    ) -> Callable[P, Self]:
        return cls._from_expr(func, name=name)
    
    @classmethod
    def from_expr_args(
        cls, 
        *args: Any | Callable[..., Self],
        name: str | None = None,
    ) -> Self:
        return cls._from_expr(*args, name=name)
    
    @property
    def expression(self) -> tuple[Callable, tuple[Any, ...], dict[str, Any]] | None:
        if self._func_args is not None:
            return self._func_args
        else:
            return None

    @property
    def mesh(self) -> Mesh | None:
        try:
            return extract_mesh(self._present)
        except ValueError:
            return None 

    @property
    def shape(self) -> tuple[int, ...]:
        return self.sequence[0].ufl_shape


class SolutionType(Protocol):
    @property
    def name(self) -> str:
        ...
    def copy(self, *args, **kwargs) -> Self:
        ...


def set_solution(solution: Function | Constant | Any, value: Any) -> None:
    return _set_solution(solution, value)

@singledispatch
def _set_solution(solution, _) -> None:
    raise MultipleDispatchTypeError(solution, set_solution)

@_set_solution.register(Function)
def _(solution: Function, value):
    if value is Unsolved:
        return set_function(solution, value.value, dofs_indices=':')
    elif isinstance(value, Function) and value.function_space == solution.function_space:
        return set_function(solution, value, dofs_indices=':')
    else:
        return set_function(solution, value)
    
@_set_solution.register(Constant)
def _(solution: Constant, value):
    if value is Unsolved:
        return set_constant(solution, value.value)
    else:
        return set_constant(solution, value)


T = TypeVar('T', bound=SolutionType)
U = TypeVar('U')
I = TypeVar('I') # TODO python 3.13 default=None
class SolutionSeries(Series[T], Generic[T, U, I]):

    @abstractmethod
    def __init__(
        self, 
        create_container: Callable[[int], T],
        name: str | tuple[str, Iterable[str]] | None,
        order: int,
        store: int | float | Callable | None = None,
        ics: I | T | U | None = None,
    ):
        super().__init__(create_container, name, order)
        self._series = []
        self._time_series = []
        self.store = store
        self._ics = None
        if ics is not None:
            self.initialize_from_ics(ics)
          
    @staticmethod
    @abstractmethod
    def set_solution(container: T, value: T | U | UnsolvedType) -> None:
        ...
    
    @Series.name.setter
    def name(self, value) -> str:
        Series.name.fset(self, value)
        for f in (*self._previous, self._present, self._future):
            f.name = self._name

    @property
    def store(self) -> int | float | Callable | None:
        return self._store

    @store.setter
    def store(self, value: int | float | Callable | None):
        self._store = value
        def _append(t: float) -> None:
            self._series.append(self._present.copy())
            self._time_series.append(t)
        if value is None:
            value = lambda: False
        self._series_append = Writer(_append, value)

    @property
    def time_series(self) -> list[float | None]:
        assert len(self._time_series) == len(self._series)
        return self._time_series
    
    @property
    def series(self) -> list[T]:
        return self._series
    
    @property
    def ics(self) -> T | None:
        return self._ics

    def initialize_from_ics(
        self,
        ics: I | T | U | None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite:
            assert self._ics is None
        self.update(ics, future=False, overwrite=overwrite)
        self._ics = self._present.copy()

    def update(
        self,
        value: T | U,
        future: bool = False,
        overwrite: bool = False,
    ):
        if future:
            container = self._future
        else:
            container = self._present

        if not overwrite:
            if not is_unsolved(container):
                raise RuntimeError('Cannot overwrite the solution if `overwrite=False`.')
        self.set_solution(container, value)

    def forward(
        self, 
        t: float | Constant | np.ndarray,
    ) -> None:
        """Steps the `Series` object forward in time.

        e.g.
        `Series([Unsolved, uⁿ⁻², uⁿ⁻¹]; uⁿ; uⁿ⁺¹) -> 
        Series([uⁿ⁻², uⁿ⁻¹, uⁿ]; uⁿ⁺¹; Unsolved)`
        """
        self._series_append.write(t)

        for i in range(-self.order + 1, 0):
            if i == -1:
                self.set_solution(self._previous[i], self._present)
            else:
                self.set_solution(self._previous[i], self._previous[i + 1])

        if not is_unsolved(self._future):
            self.set_solution(self._present, self._future)
            self.set_solution(self._future, Unsolved)
        else:
            self.set_solution(self._present, Unsolved)


class FunctionSeries(
    SolutionSeries[
        Function, 
        Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float], 
        Perturbation,
    ],
):
    def __init__(
        self,
        fs: FunctionSpace
        | tuple[Mesh, str, int]
        | tuple[Mesh, str, int, int],
        name: str | tuple[str, Iterable[str]] | None = None,
        order: int = 1,
        store: int | float | Callable[[], bool] | None = None,
        ics: Function | Perturbation| Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float] | None = None,
    ):
        fs = create_function_space(fs)
        self._function_space = fs
        self._ics_perturbation = None
        super().__init__(lambda i: Function(fs, Unsolved, index=i), name, order, store, ics)
        if self._subnames:
            assert len(self._subnames) == self._function_space.num_sub_spaces

    @staticmethod
    def set_solution(solution: Function, value):
        return set_solution(solution, value)

    @property
    def function_space(
        self,
    ) -> FunctionSpace:
        return self._function_space
    
    @property
    def mesh(
        self,
    ) -> Mesh:
        return self.function_space.mesh
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._function_space.ufl_element().value_shape()

    @property
    def dofs_series(self) -> list[np.ndarray]:
        return [i.x.array for i in self.series]

    @property
    def ics_perturbation(self) -> tuple[Function, Function] | None:
        return self._ics_perturbation

    def initialize_from_ics(
        self,
        ics,
        overwrite: bool = False,
    ) -> None:
        if isinstance(ics, Perturbation):
            self._ics_perturbation = (ics.base(self.function_space), ics.noise(self.function_space))
            ics = ics.combine_base_noise(self.function_space)
        super().initialize_from_ics(ics, overwrite)

    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> 'SubFunctionSeries':
        if name is None:
            name = self._create_subname(index)
        return SubFunctionSeries(self, index, name)
    
    def split(
        self,
        names: Iterable[str] | None = None,
    ) -> tuple['SubFunctionSeries', ...]:
        n_sub = self.function_space.num_sub_spaces
        if names is None:
            names = [None] * self.function_space.num_sub_spaces
        return tuple(self.sub(i, n) for i, n in zip(range(n_sub), names, strict=True))

        
class SubFunctionSeries(Series[Expr]):
    
    def __init__(
        self,
        mixed: FunctionSeries,
        subspace_index: int,
        name: str | None = None,
    ):
        if name is None:
            name = f'{self.name}_{subspace_index}'
        order = len(mixed.sequence) - 1
        super().__init__(
            lambda i: split(mixed.sequence[i - self.FUTURE_INDEX - 1])[subspace_index], 
            name, 
            order,
        )
        self._mixed = mixed
        self._subspace_index = subspace_index
    
    @property
    def function_space(self, collapse: bool = True) -> FunctionSpace:
        fs = self._mixed.function_space.sub(self._subspace_index)
        if collapse:
            return fs.collapse()
        else:
            return fs
    
    @property
    def mesh(self) -> Mesh:
        return self.function_space.mesh

    @property
    def series(self, collapse: bool = True) -> list[Function]:
        return [i.sub(self._subspace_index, self.name, collapse) for i in self._mixed.series]
    
    @property
    def time_series(self) -> list[float | None]:
        return self._mixed.time_series


class ConstantSeries(
    SolutionSeries[Constant, float | Iterable[float], None],
):
    def __init__(
        self,
        mesh: Mesh,
        name: str | tuple[str, Iterable[str]] | None = None,
        order: int = 1,
        shape: tuple[int, ...] = (),
        store: int | float | Callable[[], bool] | None = None,
        ics: Constant | float | Iterable[float] | None = None,
    ):
        super().__init__(lambda i: Constant(mesh, Unsolved, shape=shape, index=i), name, order, store, ics)
        self._mesh = mesh
        self._shape = shape
        if self._subnames:
            if self.shape == ():
                raise SubSeriesError
            assert len(self._subnames) == self.shape[0]

    @staticmethod
    def set_solution(container: Constant, value):
        return set_solution(container, value)

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def value_series(self) -> list[float | np.ndarray]:
        return [
            i.value.item() if i.value.shape == () else i.value
            for i in self.series
        ]
    
    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self:
        if self.shape == ():
            raise SubSeriesError

        if name is None:
            name = self._create_subname(index)
        
        subseries = ConstantSeries(self.mesh, name, self.order, self.shape[1:], store=1)
        for c, t in zip(self.series, self.time_series):
            subseries.update(c.value[index])
            subseries.forward(t)
        subseries.store = self.store

        return subseries
    
    def split(
        self,
        names: Iterable[str] | None = None,
    ) -> tuple[Self, ...]:
        if self.shape == ():
            raise SubSeriesError
        subseries_indices = tuple(range(self.shape[0]))
        if names is None:
            names = [None] * self.shape[0]
        return tuple(self.sub(i, n) for i, n in zip(subseries_indices, names, strict=True))
    

class SubSeriesError(RuntimeError):
    def __init__(self):
        super().__init__('Scalar-valued series cannot have a subseries.')


