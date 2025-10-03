from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, TypeVar, Iterable, Generic, Protocol, ParamSpec
from typing_extensions import Self

import numpy as np
from ufl import TestFunction, TrialFunction, split
from ufl.core.expr import Expr
from dolfinx.fem import FunctionSpace, Function, Constant, Expression
from dolfinx.mesh import Mesh

from ..utils import set_fem_constant, set_fem_function, extract_mesh, fem_function_space, grid
from ..utils.fem_utils import ScalarVectorError
from ..utils.fem_typecasting import fem_function_components
from ..utils.deferred import Writer
from ..utils.fem_perturbation import Perturbation
from ..fem import LUCiFExFunction, LUCiFExConstant, Unsolved, UnsolvedType, is_unsolved


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
        name: str | None,
        order: int,
    ):
        assert order >= 1
        self._future = create_container(self.FUTURE_INDEX)
        self._present = create_container(self.FUTURE_INDEX - 1)
        self._previous = [create_container(i) for i in range(-1, -order, -1)][::-1]
        if name is None:
            name = self.__class__.__name__
        self.name = name
    
    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """Time-independent mesh that the series is defined on."""
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

    def __repr__(self) -> str:
        seq = [i if not is_unsolved(i) else Unsolved for i in self.sequence]
        seq_str = [str(s) for s in seq]
        previous = ', '.join(seq_str[:-2])
        present, future = seq_str[-2:]
        if previous:
            seq_repr = f"{previous}; {present}; {future}"
        else:
            seq_repr = f"{present}; {future}"
        return f"{self.__class__.__name__}({seq_repr})"

    def __str__(self) -> str:
        return f"{self.name}"

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
        else:
            seq = [getattr(i, method)(other) for i in self]
        return ExprSeries(seq)


P = ParamSpec('P')
class ExprSeries(
    Series[Expr | Function | Constant],
):
    
    _func = None
    _func_args = None

    def __init__(
        self,
        arg: Iterable[Expr | Function | Constant] | Self,
        name: str | None = None,
    ):
        if isinstance(arg, Series):
            if name is None:
                name = arg.name
            self.__init__(arg.sequence, name)
        else:
            order = len(arg) - 1
            super().__init__(lambda i: arg[i - self.FUTURE_INDEX - 1], name, order)

    @classmethod
    def from_relation(
        cls, 
        func: Callable[P, Self], 
        name: str | None = None,
    ) -> Callable[P, Self]:
        def _(*args: P.args, **kwargs: P.kwargs):
            if kwargs:
                raise NotImplementedError('Provide positional arguments only.')
            expr = func(*args, **kwargs)
            obj = cls(expr, name)
            obj._func = func
            obj._func_args = args
            return obj
        return _

    @property
    def relation(self) -> tuple[Callable, tuple] | None:
        if self._func is not None:
            return self._func, self._func_args
        else:
            return None

    @property
    def mesh(self) -> Mesh:
        return extract_mesh(self._present)
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self.sequence[0].ufl_shape


class ContainerType(Protocol):
    @property
    def name(self) -> str:
        ...
    def copy(self, *args, **kwargs) -> Self:
        ...


T = TypeVar('T', bound=ContainerType)
U = TypeVar('U')
I = TypeVar('I') # TODO python 3.13 default=None
class ContainerSeries(Series[T], Generic[T, U, I]):

    @abstractmethod
    def __init__(
        self, 
        create_container: Callable[[int], T],
        name: str | None,
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
    def _set_container(container: T, value: T | U | UnsolvedType) -> None:
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
            assert is_unsolved(container)
        self._set_container(container, value)

    def forward(
        self, 
        t: float | Constant | np.ndarray,
    ) -> None:
        """Steps the `Series` object forward in time.

        e.g.
        `([Unsolved, u₋₂, u₋₁]; u₀; u₁) -> ([u₋₂, u₋₁, u₀]; u₁; Unsolved)`
        """
        self._series_append.write(t)

        for i in range(-self.order + 1, 0):
            if i == -1:
                self._set_container(self._previous[i], self._present)
            else:
                self._set_container(self._previous[i], self._previous[i + 1])

        if not is_unsolved(self._future):
            self._set_container(self._present, self._future)
            self._set_container(self._future, Unsolved)
        else:
            self._set_container(self._present, Unsolved)


class FunctionSeries(
    ContainerSeries[
        LUCiFExFunction, 
        Function | Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float], 
        Perturbation,
    ],
):

    def __init__(
        self,
        function_space: FunctionSpace
        | tuple[Mesh, str, int]
        | tuple[Mesh, str, int, int],
        name: str | None = None,
        order: int = 1,
        store: int | float | Callable[[], bool] | None = None,
        ics: Function | Perturbation| Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float] | None = None,
    ):
        function_space = fem_function_space(function_space)
        self._function_space = function_space
        self._ics_perturbation = None
        super().__init__(lambda i: LUCiFExFunction(function_space, Unsolved, index=i), name, order, store, ics)

    @staticmethod
    def _set_container(container: LUCiFExFunction, value):
        if value is Unsolved:
            return set_fem_function(container, value.value, dofs_indices=':')
        elif isinstance(value, LUCiFExFunction) and value.function_space == container.function_space:
            return set_fem_function(container, value, dofs_indices=':')
        else:
            return set_fem_function(container, value)

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
    def ics_perturbation(self) -> tuple[LUCiFExFunction, LUCiFExFunction] | None:
        return self._ics_perturbation

    @property
    def dofs_series(self) -> list[np.ndarray]:
        return [i.x.array for i in self.series]

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
        subspace_index: int, 
        name: str | None,
    ) -> 'SubFunctionSeries':
        if name is None:
            name = f'{self.name}_{subspace_index}'
        return SubFunctionSeries(self, subspace_index, name)
    
    def split(
        self,
        names: Iterable[str] | None,
    ) -> tuple['SubFunctionSeries', ...]:
        subspace_indices = tuple(range(self.function_space.num_sub_spaces))
        if names is None:
            names = [f'{self.name}_{i}' for i in subspace_indices]
        return tuple(SubFunctionSeries(self, i, n) for i, n in zip(subspace_indices, names, strict=True))


class ConstantSeries(
    ContainerSeries[LUCiFExConstant, Constant | float | Iterable[float], None],
):
    def __init__(
        self,
        mesh: Mesh,
        name: str | None = None,
        order: int = 1,
        shape: tuple[int, ...] = (),
        store: int | float | Callable[[], bool] | None = None,
        ics: LUCiFExConstant | float | Iterable[float] | None = None,
    ):
        super().__init__(lambda i: LUCiFExConstant(mesh, Unsolved, shape=shape, index=i), name, order, store, ics)
        self._mesh = mesh
        self._shape = shape

    @staticmethod
    def _set_container(container: LUCiFExConstant, value):
        if value is Unsolved:
            return set_fem_constant(container, value.value)
        else:
            return set_fem_constant(container, value)

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
    def series(self, collapse: bool = True) -> list[LUCiFExFunction]:
        return [i.sub(self._subspace_index, self.name, collapse) for i in self._mixed.series]
    
    @property
    def time_series(self) -> list[float | None]:
        return self._mixed.time_series


class NumericSeries:
    def __init__(
        self, 
        series: Iterable[float | int | np.ndarray], 
        t: Iterable[float] | np.ndarray,
        name: str | None = None,
    ): 
        assert len(series) == len(t)
        self._series = list(series)
        self._time_series = list(t)
        if name is None:
            name = self.__class__.__name__
        self.name = name

    @classmethod
    def from_series(
        cls, 
        u: ConstantSeries,
    ) -> Self:
        return cls(u.value_series, u.time_series, u.name)
    
    @property
    def series(self) -> list[float | int | np.ndarray]:
        return self._series

    @property
    def time_series(self) -> list[float]:
        return self._time_series
    
    @property
    def name(self) -> str | None:
        return self._name
    
    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        assert '__' not in value
        self._name = value
    
    @cached_property
    def shape(self) -> tuple[int, ...] | list[tuple[int, ...]] | None:
        if not self.series:
            return None
        get_shape = lambda x: () if isinstance(x, (float, int)) else x.shape
        shapes = [get_shape(i) for i in self.series]
        if len(set(shapes)) == 1:
            return shapes[0]
        else:
            return shapes
        
    @property
    def is_homogeneous(self) -> bool:
        return not isinstance(self.shape, list)

    
class GridSeries(NumericSeries):
    def __init__(
        self, 
        series: list[np.ndarray], 
        t: list[float], 
        x: tuple[np.ndarray, ...],
        name: str | None = None,
    ): 
        super().__init__(series, t, name)
        self._axes = x

    @property
    def series(self) -> list[np.ndarray]:
        return self._series
    
    @classmethod
    def from_series(
        cls, 
        u: FunctionSeries,
        strict: bool = False,
        jit: bool | None = None,
    ) -> Self:
        match u.shape:
            case ():
                series = [grid(i, strict, jit) for i in u.series]
            case (dim, ):
                uxyz_series = [fem_function_components(('P', 1), i) for i in u.series]
                series = [np.array([grid(j) for j in i[:dim]]) for i in uxyz_series]
            case _:
                raise ScalarVectorError(u)
            
        return cls(
            series,
            u.time_series,
            grid(u.mesh, strict), 
            u.name,
        )
    
    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        return self._axes
    



# class StaticExpr(
#     Expr, 
#     ContainerType,
#     metaclass=type('', (type(ContainerType), type(Expr)), {}),
# ):
#     pass


# def expr_container(
#     expr: Expr,
#     name: str | None,
# ) -> StaticExpr:
#     if name is None:
#         name = expr.__class__.__name__

#     class _StaticExpr(expr.__class__):
        
#         def __init__(self, *operands):
#             super().__init__(*operands)
#             self._name = name

#         @property
#         def name(self):
#             return self._name
        
#         def copy(self):
#             print('copying!!!') # TODO
#             raise NotImplementedError

#     return _StaticExpr(*expr.ufl_operands)


    
# @classmethod
# def from_sequence(
#     cls, 
#     sequence: Iterable[Expr | Iterable[Function | Constant] | None],
#     name: str | None = None,
# ) -> Self:

#     past = sequence[:-2]
#     present = sequence[-2]
#     future = sequence[-1]

#     order = len(sequence) - 1
#     expr_series = cls(present, name, order)

#     # for value in past:
#     #     expr_series.update(value)
#     #     expr_series.forward(count=False)

#     # expr_series.update(present, future=False)
#     # expr_series.update(future, future=True)

#     expr_series._previous = sequence[:-2]
#     expr_series._present = sequence[-2]
#     expr_series._future = sequence[-1]
#     return expr_series

# def _create_container(e: Expr, i: int) -> Expr:
# coeffs_consts = (*extract_coefficients(e), *extract_constants(e))
# mapping = {}
# for u in coeffs_consts:
#     if isinstance(u, Function):
#         mapping[u] = StaticFunction(u.function_space, Unsolved, u.name, index=i)
#     elif isinstance(u, StaticConstant):
#         mapping[u] = StaticConstant(u.mesh, Unsolved, u.name, u.ufl_shape, i)
#     else:
#         raise MultipleDispatchTypeError(u)

# return expr_container(replace(sequence, mapping), name)

# @staticmethod
# def _set_container(
#     container: StaticExpr, 
#     value,
# ):
#     coeffs_consts =(*extract_coefficients(container), *extract_constants(container))
#     if value is Unsolved:
#         value = [Unsolved] * len(coeffs_consts)
#     if isinstance(value, Expr):
#         value = [*extract_coefficients(value), *extract_constants(value)]
#     # TODO what is correct order of `value: tuple`?
#     for c, v in zip(coeffs_consts, value, strict=True):
#         if isinstance(c, StaticFunction):
#             FunctionSeries._set_container(c, v)
#         elif isinstance(c, StaticConstant):
#             ConstantSeries._set_container(c, v)
#         else:
#             raise TypeError

# class old_ExprSeries(
#     Series[Terminal, Iterable[StaticFunction | StaticConstant]],
# ):

#     def __init__(
#         self, 
#         name: str | None = None, 
#         order: int = 1, 
#         store = None, 
#         ics = None,
#     ):
#         # TODO how best to create unsolved UFL expression?
#         def _create_unsolved():
#             unsolved = Terminal()
#             return unsolved

#         super().__init__(_create_unsolved, name, order, store, ics)

# @classmethod
# def from_sequence(
#     cls, 
#     sequence: Iterable[Expr | UnsolvedType | None],
# ) -> Self:
#     order = len(sequence) - 1
#     obj = cls(order=order)
#     seq = [
#         obj._create_unsolved() if i is None or i is Unsolved else i
#         for i in sequence
#     ]
#     obj._previous = seq[:-2]
#     obj._present = seq[-2]
#     obj._future = seq[-1]
#     return obj
