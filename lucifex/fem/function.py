from typing_extensions import Self
from collections.abc import Callable

import numpy as np
from dolfinx.mesh import Mesh
from dolfinx.fem import FunctionSpace, Expression
from dolfinx.fem import Function as DOLFINxFunction
from dolfinx.la import VectorMetaClass
from petsc4py import PETSc

from ..utils import function_space, SpatialPerturbation
from .unsolved import UnsolvedType


class Function(DOLFINxFunction):
    """
    Subclass with additional utilities, not to be confused with `dolfinx.fem.Function.`
    """
    def __init__(
        self,
        fs: FunctionSpace
            | tuple[Mesh, str, int]
            | tuple[Mesh, str, int, int],
        x: VectorMetaClass
            | Self
            | Expression
            | Callable[[np.ndarray], np.ndarray]
            | SpatialPerturbation
            | float
            | UnsolvedType
            | None
        = None,
        name: str | None = None,
        dtype: np.dtype = PETSc.ScalarType,
        index: int | None = None,
    ):
        fs = function_space(fs)

        if name is None:
            name = f'f{id(self)}'

        if isinstance(x, (DOLFINxFunction, Expression, Callable, SpatialPerturbation, int, float, UnsolvedType)):
            super().__init__(fs, None, name, dtype)
            if isinstance(x, SpatialPerturbation):
                x = x.combine_base_noise(fs)
            if isinstance(x, UnsolvedType):
                x = x.value
            if isinstance(x, (int, float)):
                self.x.array[:] = float(x)
            else:
                self.interpolate(x)
        else:
            super().__init__(fs, x, name, dtype)
        
        self._index = index

    def copy(self, name: str | None = None) -> Self:
        f = super().copy()
        if name is None:
            f.name = self.name
        else:
            f.name = name
        return f
    
    @property
    def index(self) -> int | None:
        return self._index

    def __str__(self) -> str:
        name = super().__str__()
        if self.index is None:
            return name
        else:
            return unicode_superscript(name, self.index)
        
    def sub(
        self, 
        subspace_index: int, 
        name: str | None = None,
        collapse: bool = False,
    ) -> Self:
        if name is None:
            name = f'{self.name}_{subspace_index}'
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
        subspace_indices = tuple(range(self.function_space.num_sub_spaces))
        if names is None:
            names = [f'{self.name}_{i}' for i in subspace_indices]
        return tuple(self.sub(i, n, collapse) for i, n in zip(subspace_indices, names, strict=True))
        
    def collapse(self) -> Self:
        u = self._cpp_object.collapse()
        fs = FunctionSpace(None, self.ufl_element(), u.function_space)
        return Function(fs, u.x, self.name, index=self.index)
        

def unicode_superscript(name: str, n: int) -> str:
    digits_supers = {
        '0': '⁰', 
        '1': '¹', 
        '2': '²', 
        '3': '³', 
        '4': '⁴', 
        '5': '⁵', 
        '6': '⁶',
        '7': '⁷', 
        '8': '⁸', 
        '9': '⁹',
    }
    n_str = str(n)
    if '-' in n_str:
        n_str = n_str.replace('-', '')
        
    superscript = ''
    for _s in n_str:
        superscript = f'{superscript}{digits_supers[_s]}'

    if n < 0:
        superscript = f'⁻{superscript}'
    if n > 0:
        superscript = f'⁺{superscript}'

    return f'{name}⁽{superscript}⁾'

