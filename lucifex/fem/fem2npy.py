from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Iterable
from typing_extensions import Self

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh

from ..mesh.mesh2npy import (
    NPyMesh, GridMesh, TriMesh, QuadMesh, as_tri_mesh, as_grid_mesh,
    as_npy_object,
)
from ..utils.py_utils import optional_lru_cache, replicate_callable
from ..utils.fenicsx_utils import (
    dofs, grid_values, get_component_functions,
    NonScalarVectorError, extract_mesh,
)
from .function import Function
from .constant import Constant


class NPyNameAttr(ABC):
    
    def __init__(
        self,
        name: str | tuple[str, Iterable[str]] | None = None,
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
    ) -> tuple[Self, ...] | None:
        if self.ufl_shape == ():
            return None
        else:
            n_sub = self.ufl_shape[0]
        if names is None:
            names = [self._create_subname(i) for i in range(n_sub)]
        return tuple(self.sub(i, n) for i, n in zip(range(n_sub), names, strict=True))



T = TypeVar('T')
class NPyNameValueAttr(NPyNameAttr, Generic[T]):
    
    def __init__(
        self,
        value: T,
        name: str | None = None,
    ):
        super().__init__(name)
        self._value = value

    @property
    def value(self) -> T: #FIXME np.ndarray shape type hints
        return self._value


class NPyConstant(NPyNameValueAttr[float | int | np.ndarray]):
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


M = TypeVar('M', bound=NPyMesh)
class NPyFunction(NPyNameValueAttr[np.ndarray | tuple[np.ndarray, ...]], Generic[M]):
    def __init__(
        self,
        value: np.ndarray | tuple[np.ndarray, ...],
        mesh: M,
        name: str | None = None,
    ):
        super().__init__(value, name)
        self._mesh = mesh
    
    @property
    def mesh(self) -> M:
        return self._mesh
    
    @property
    def ufl_shape(self) -> tuple[int, ...]:
        if isinstance(self._value, tuple):
            return (len(self._value), )
        else:
            return ()
        
    @property 
    def npy_shape(self) -> tuple[int, ...]:
        if self.ufl_shape == ():
            return self._value.shape
        else:
            return self._value[0].shape
        
    def sub(
        self: 'NPyFunction',
        index: int,
        name: str | None = None,
    ) -> Self | None:
        if self.ufl_shape == ():
            return None
        if name is None:
            name = self._create_subname(index)
        return self.__class__(self._value[index], self.mesh, name)
    
    @classmethod
    @abstractmethod
    def from_function(
        cls,
        u: Function | Expr,
        get_values: Callable[[Function | Expr], np.ndarray],
        convert_mesh: Callable[[Mesh], M] | Mesh,
    ):
        match u.ufl_shape:
            case (_, ):
                values = tuple(
                    get_values(ui) for ui in get_component_functions(('P', 1), u, use_cache=Ellipsis)
                )
            case ():
                values = get_values(u)
            case _:
                raise NonScalarVectorError(u)
            
        msh = convert_mesh(extract_mesh(u))
        return cls(
            values,
            msh,
            u.name if isinstance(u, Function) else None, #FIXME
        )
    

class GridFunction(NPyFunction[GridMesh]):
    @classmethod
    def from_function(
        cls: type['GridFunction'],
        u: Function | Expr,
        strict: bool = False,
        jit: bool = True,
        mask: float = np.nan,
        use_mesh_map: bool = False,
        use_mesh_cache: bool = True,
        mesh: Mesh | None = None,
    ) -> Self:
        get_values = lambda u: grid_values(u, strict, jit, mask, use_mesh_map, use_mesh_cache, mesh)
        convert_mesh = lambda m: as_grid_mesh(use_cache=use_mesh_cache)(m, strict)
        return super().from_function(
            u,
            get_values,
            convert_mesh,
        )
    

class TriFunction(NPyFunction[TriMesh]):
    @classmethod
    def from_function(
        cls: type['TriFunction'],
        u: Function | Expr,
        use_mesh_cache: bool = True,
        mesh: Mesh | None = None,
    ) -> Self:
        if mesh is None:
            elem = ('P', 1)
        else:
            elem = (mesh, 'P', 1)
        get_values = lambda u: dofs(u, elem, try_identity=True)
        convert_mesh = lambda m: as_tri_mesh(use_cache=use_mesh_cache)(m)
        return super().from_function(
            u,
            get_values,
            convert_mesh,
        )
    

class QuadFunction(NPyFunction[QuadMesh]):
    @classmethod
    def from_function(
        cls: type['QuadFunction'],
        u: Function | Expr,
        use_mesh_cache: bool = True,
        mesh: Mesh | None = None,
    ) -> Self:
        raise NotImplementedError


as_npy_constant = replicate_callable(NPyConstant.from_constant)(lambda: None)

as_grid_function = optional_lru_cache(
    replicate_callable(GridFunction.from_function)(lambda: None)
)

as_tri_function = optional_lru_cache(
    replicate_callable(TriFunction.from_function)(lambda: None)
)

as_quad_function = optional_lru_cache(
    replicate_callable(QuadFunction.from_function)(lambda: None)
)


def as_npy_function(
    u: Function | Expr,
    grid: bool | None = None,
    use_cache: bool | tuple[bool, bool] = True,
    mesh: Mesh | None = None,
):
    if isinstance(use_cache, bool):
        use_cache = use_cache, use_cache
    use_mesh_cache, use_func_cache = use_cache

    if mesh is None:
        mesh = extract_mesh(u)

    return as_npy_object(
        u,
        as_grid_function(use_cache=use_func_cache),
        as_tri_function(use_cache=use_func_cache),
        as_quad_function(use_cache=use_func_cache),
        mesh,
        grid,
        use_mesh_cache,
    )