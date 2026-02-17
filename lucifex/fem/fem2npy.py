from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Iterable
from typing_extensions import Self

import numpy as np
from dolfinx.mesh import Mesh

from ..mesh.mesh2npy import NPyMesh, GridMesh, TriMesh, QuadMesh, as_tri_mesh, as_grid_mesh
from ..utils.py_utils import optional_lru_cache, replicate_callable
from ..utils.fenicsx_utils import (
    dofs, grid_dofs, is_grid, is_simplicial, get_component_functions,
    NonScalarVectorError, NonCartesianQuadMeshError,
)
from .function import Function
from .constant import Constant


class NPyNamedObject(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str | None = None,
    ):
        self._name = name
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
    def name(self) -> str | None:
        return self._name
    
    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        assert '__' not in value
        self._name = value

    @property
    @abstractmethod
    def ufl_shape(self) -> tuple[int, ...] | None:
        ...

    @abstractmethod
    def sub(
        self,
        index: int,
        name: str | None = None,
    ) -> Self | None:
        ...

    def split(
        self,
        names: Iterable[str] | None = None,
    ) -> Self | None:
        if self.ufl_shape == ():
            return None
        else:
            n_sub = self.ufl_shape[0]
        if names is None:
            names = [self._create_subname(i) for i in range(n_sub)]
        return tuple(self.sub(i, n) for i, n in zip(range(n_sub), names, strict=True))


M = TypeVar('M', bound=NPyMesh)
class NPyFunction(NPyNamedObject, Generic[M]):
    def __init__(
        self,
        values: np.ndarray | tuple[np.ndarray, ...],
        mesh: M,
        name: str | None = None,
    ):
        super().__init__(name)
        self._values = values
        self._mesh = mesh

    @property
    def values(self) -> np.ndarray: #FIXME shape type hints
        return self._values
    
    @property
    def mesh(self) -> M:
        return self._mesh
    
    @property
    def ufl_shape(self) -> tuple[int, ...]:
        if isinstance(self._values, tuple):
            return (len(self._values), )
        else:
            return ()
        
    @property 
    def npy_shape(self) -> tuple[int, ...]:
        if self.ufl_shape == ():
            return self._values.shape
        else:
            return self._values[0].shape
        
    def sub(
        self: 'NPyFunction',
        index: int,
        name: str | None = None,
    ) -> Self | None:
        if self.ufl_shape == ():
            return None
        if name is None:
            name = self._create_subname(index)
        return self.__class__(self._values[index], self.mesh, name)
    
    @classmethod
    @abstractmethod
    def from_function(
        cls,
        u: Function,
        values_func: Callable[[Function], np.ndarray],
        mesh_func: Callable[[Mesh], M],
    ):
        match u.ufl_shape:
            case (_, ):
                values = tuple(
                    values_func(ui) for ui in get_component_functions(('P', 1), u, use_cache=Ellipsis)
                )
            case ():
                values = values_func(u)
            case _:
                raise NonScalarVectorError(u)
            
        msh = mesh_func(u.function_space.mesh)
        return cls(
            values,
            msh,
            u.name,
        )
    

class GridFunction(NPyFunction[GridMesh]):
    @classmethod
    def from_function(
        cls: type['GridFunction'],
        u: Function,
        strict: bool = False,
        jit: bool = True,
        mask: float = np.nan,
        use_mesh_map: bool = False,
        use_mesh_cache: bool = True,
    ) -> Self:
        values_func = lambda u: grid_dofs(u, strict, jit, mask, use_mesh_map, use_mesh_cache)
        mesh_func = lambda m: as_grid_mesh(use_cache=use_mesh_cache)(m, strict)
        return super().from_function(
            u,
            values_func,
            mesh_func,
        )
    

class TriFunction(NPyFunction[TriMesh]):
    @classmethod
    def from_function(
        cls: type['TriFunction'],
        u: Function,
        use_mesh_cache: bool = True,
    ) -> Self:
        """
        Interpolates function to P₁ (which has identity vertex-to-DoF map)
        to evaluate the function at the vertex values.

        Note that this is suitable on both Cartesian and unstructured meshes.
        """
        values_func = lambda u: dofs(u, ('P', 1), try_identity=True)
        mesh_func = lambda m: as_tri_mesh(use_cache=use_mesh_cache)(m)
        return super().from_function(
            u,
            values_func,
            mesh_func,
        )
    

class QuadFunction(NPyFunction[QuadMesh]):
    ...


class NPyConstant(NPyNamedObject):
    def __init__(
        self,
        value: float | int | np.ndarray,
        name: str | None = None,
    ):
        super().__init__(name)
        self._value = value

    @classmethod
    def from_constant(
        cls: type['NPyConstant'],
        c: Constant,
    ) -> Self:
        return cls(c.value, c.name)
    
    @property
    def value(self):
        return self._value
    
    @property
    def ufl_shape(self) -> tuple[int, ...]:
        if isinstance(self._value, (float, int)):
            return ()
        return self._value.shape
    
    def sub(
        self: 'NPyConstant',
        index: int,
        name: str | None = None,
    ) -> Self | None:
        if self.ufl_shape == ():
            return None
        return self.__class__(
            self._value[index], 
            self._create_subname(index) if name is None else name,
        )


as_grid_function = optional_lru_cache(
    replicate_callable(GridFunction.from_function)(lambda: None)
)
as_tri_function = optional_lru_cache(
    replicate_callable(TriFunction.from_function)(lambda: None)
)

as_npy_constant = replicate_callable(NPyConstant.from_constant)(lambda: None)


def as_npy_function(
    u: Function,
    cartesian: bool | None = None,
    use_cache: bool | tuple[bool, bool] = (True, False),
) -> GridFunction | TriFunction:
    
    mesh = u.function_space.mesh

    if isinstance(use_cache, bool):
        use_cache = (use_cache, use_cache)
    use_mesh_cache, use_func_cache = use_cache

    if cartesian is None:
        cartesian = is_grid(use_cache=use_mesh_cache)(mesh)
    simplicial = is_simplicial(use_cache=use_mesh_cache)(mesh)

    match simplicial, cartesian:
        case True, False:
            return as_tri_function(use_cache=use_func_cache)(u)
        case _, True:
            return as_grid_function(use_cache=use_func_cache)(u)
        case False, False:
            raise NonCartesianQuadMeshError







