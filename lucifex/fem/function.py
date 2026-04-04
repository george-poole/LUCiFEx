from typing import Iterable
from typing_extensions import Self
from collections.abc import Callable
from functools import cached_property

import numpy as np
from dolfinx.mesh import Mesh
from dolfinx.fem import FunctionSpace, Expression
from dolfinx.fem import Function as DOLFINxFunction
from dolfinx.la import VectorMetaClass
from petsc4py import PETSc

from ..utils.fenicsx_utils import create_function_space, set_function, extract_subspaces
from ..utils.py_utils import str_indexed
from ..utils.npy_utils import AnyNumber
from .perturbation import SpatialPerturbation
from .unsolved import UnsolvedType


class Function(DOLFINxFunction):

    def __new__(cls, *args, **kwargs) -> Self:
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        fs: FunctionSpace
            | tuple[Mesh, str, int]
            | tuple[Mesh, str, int, int],
        value: VectorMetaClass
            | Self
            | Expression
            | Callable[[np.ndarray], np.ndarray]
            | SpatialPerturbation
            | AnyNumber
            | UnsolvedType
            | None
        = None,
        name: str | tuple[str, Iterable[str]] | None = None,
        dtype: np.dtype = PETSc.ScalarType,
        index: int | None = None,
    ):
        """
        Subclass with additional utilities, not to be confused with `dolfinx.fem.Function.`
        """
        fs = create_function_space(fs)

        if isinstance(name, tuple):
            name, subnames = name
            subnames = tuple(subnames)
        else:
            subnames = None
        if name is None:
            name = f'{self.__class__.__name__}{id(self)}'

        if isinstance(
            value, 
            (
                self.__class__,
                DOLFINxFunction, 
                Expression, 
                Callable, 
                SpatialPerturbation, 
                AnyNumber, 
                UnsolvedType,
            ),
        ):
            super().__init__(fs, None, name, dtype)
            if isinstance(value, (AnyNumber, UnsolvedType)):
                self.x.array[:] = float(value)
            else:
                if isinstance(value, SpatialPerturbation):
                    value = value.combine_base_noise(fs)
                # self.interpolate(value)
                set_function(self, value)
        else:
            super().__init__(fs, value, name, dtype)

        self._subnames = subnames
        self._create_subname = lambda i: (
            self._subnames[i] if self._subnames else f'{self.name}{i}'
        )
        self._index = index

    def copy(self, name: str | None = None) -> Self:
        if name is None:
            name = self.name
        return Function(
            self.function_space,
            type(self.x)(self.x),
            name=name,
        )
    
    @property
    def index(self) -> int | None:
        return self._index

    def __str__(self) -> str:
        name = super().__str__()
        if self.index is None:
            return name
        else:
            return str_indexed(name, self.index, 'superscript', True)
        
    @cached_property
    def function_subspaces(
        self,
    ) -> tuple[FunctionSpace, ...]:
        """
        Cached collapsed subspaces of the function space.
        """
        return extract_subspaces(self.function_space)
        
    def sub(
        self, 
        subspace_index: int, 
        name: str | None = None,
        collapse: bool = False,
    ) -> Self:
        if name is None:
            name = self._create_subname(subspace_index)
        
        f = Function(
            self.function_space.sub(subspace_index), 
            self.x, 
            name, 
            index=self.index,
        )
        if collapse:
            return f.collapse()
        else:
            return f
        
    def split(
        self,
        names: tuple[str, ...] | None = None,
        collapse: bool = False,
    ) -> tuple[Self, ...]:
        num_sub_spaces = self.function_space.num_sub_spaces
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        if names is None:
            names = [None] * self.function_space.num_sub_spaces
        subspace_indices = tuple(range(self.function_space.num_sub_spaces))
        return tuple(self.sub(i, n, collapse) for i, n in zip(subspace_indices, names, strict=True))
        
    def collapse(self) -> Self:
        u = self._cpp_object.collapse()
        fs = FunctionSpace(None, self.ufl_element(), u.function_space)
        return Function(fs, u.x, self.name, index=self.index)

