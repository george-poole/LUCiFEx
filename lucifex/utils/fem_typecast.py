from collections.abc import Callable, Iterable
from typing import overload
from types import EllipsisType
from functools import singledispatch, lru_cache

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, FunctionSpace, Constant, Function, VectorFunctionSpace
from dolfinx.fem import Expression
from ufl import FiniteElement, MixedElement, VectorElement

from .fem_utils import (
    is_same_element, function_subspace,
    extract_mesh, is_vector, is_component_space, VectorError
)
from .py_utils import MultipleDispatchTypeError, StrSlice, optional_lru_cache
from .fem_mutate import set_finite_element_function


@overload
def function_space(
    fs: FunctionSpace,
    subspace_index: int | None = None,
) -> FunctionSpace:
    ...


@overload
def function_space(
    fs: tuple[Mesh, str, int]
    | tuple[Mesh, str, int]
    | tuple[Mesh, str, int, int]
    | tuple[Mesh, Iterable[tuple[str, int] | tuple[str, int, int]]],
    subspace_index: int | None = None,
    use_cache: bool = False,
) -> FunctionSpace:
    ...


def function_space(
    fs,
    subspace_index=None,
    use_cache=False,
) -> FunctionSpace:
    """
    Typecast to `dolfinx.fem.FunctionSpace`.
    """
    if isinstance(fs, FunctionSpace):
        return function_subspace(fs, subspace_index)
    else:
        return _function_space(use_cache=use_cache)(fs, subspace_index)
    

@optional_lru_cache
def _function_space(
    fs_hashable: tuple[Mesh, str, int]
    | tuple[Mesh, str, int]
    | tuple[Mesh, str, int, int]
    | tuple[Mesh, Iterable[tuple[str, int] | tuple[str, int, int]]],
    subspace_index: int | None = None,
) -> FunctionSpace:
    match fs_hashable:
        case mesh, elements:
            mixed_elements = []
            for e in elements:
                if len(e) == 2:
                    fam, deg = e
                    mixed_elements.append(FiniteElement(fam, mesh.ufl_cell(), deg))
                elif len(e) == 3:
                    fam, deg, dim = e
                    mixed_elements.append(VectorElement(fam, mesh.ufl_cell(), deg, dim))
                else:
                    raise ValueError(f'{e}')
            fs = FunctionSpace(mesh, MixedElement(*mixed_elements))

        case mesh, fam, deg:
            fs = FunctionSpace(mesh, (fam, deg))
        
        case mesh, fam, deg, dim:
            fs = VectorFunctionSpace(mesh, (fam, deg), dim)
        
        case _:
            raise ValueError(f'{fs_hashable}')

    return function_subspace(fs, subspace_index)


def create_function(
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[Mesh, str, int, int] | tuple[str, int] | tuple[str, int, int],
    value: Function | Constant | Expression | Expr | Callable[[np.ndarray], np.ndarray] | float | Iterable[float | Callable[[np.ndarray], np.ndarray]],
    subspace_index: int | None = None,
    dofs_indices: Iterable[int] | StrSlice | None = None,
    name: str | None = None,
    try_identity: bool = False,
    use_cache: bool | EllipsisType = False,
) -> Function:
    """
    Typecast to `dolfinx.fem.Function`. 

    `use_cache = True` will try to fetch a `Function` from the cache, based on the tuple-valued `fs`, 
    `subspace_index` and `name` and then mutate that `Function`. \\
    `use_cache = Ellipsis` will try to fetch a `FunctionSpace` from the cache, but create a new `Function`. \\
    `use_cache = False` will create a new `FunctionSpace` and a new `Function`.
    """
    if name is None:
        try:
            name = value.name
        except AttributeError:
            pass
            
    if isinstance(fs, FunctionSpace):
        if try_identity and isinstance(value, Function) and value.function_space == fs:
            return value
        fs = function_subspace(fs, subspace_index)
        f = Function(fs, name=name)
    else:
        fs = _as_function_space_tuple(fs, value)
        if try_identity and isinstance(value, Function) and is_same_element(value, *fs[1:], mesh=fs[0]):
            return value
        if use_cache is True:
            f = _create_function(fs, subspace_index, name)
        else:
            fs = function_space(fs, subspace_index, bool(use_cache))
            f = Function(fs, name=name)

    set_finite_element_function(f, value, dofs_indices)
    return f


@lru_cache
def _create_function(
    fs_hashable: tuple, 
    subspace_index: int | None,
    name: str | None,
) -> Function:
    fs = _function_space(use_cache=True)(fs_hashable, subspace_index)
    return Function(fs, name=name)


@optional_lru_cache
def get_subfunctions(
    u: Function,  
    collapse: bool = True,
) -> tuple[Function, ...]:
    if collapse:
        return tuple(i.collapse() for i in u.split())
    else:
        return u.split()


def get_component_functions(
    fs: tuple[Mesh, str, int] | tuple[str, int],
    u: Function | Expr,
    names: Iterable[str | None] | None = None,
    use_cache: bool | EllipsisType = False,
) -> tuple[Function, ...]:
    
    if not is_vector(u):
        raise VectorError(u)
    
    if isinstance(fs, tuple):
        fs = _as_function_space_tuple(fs, u)

    dim = u.ufl_shape[0]
    
    if names is None:
        try:
            u_name = u.name
        except AttributeError:
            u_name = f'{u.__class__.__name__}{id(u)}'
        names = tuple(f'{u_name}_{i}' for i in range(dim))

    if isinstance(u, Function) and is_component_space(u.function_space):
        # e.g. `u(𝐱) ∈ P₁⨯P₁`
        return tuple(
            create_function(fs, i.collapse(), name=n, use_cache=use_cache) 
            for i, n in zip(u.split(), names, strict=True)
        )
    else:
        # e.g. `u(𝐱) ∈ BDM`
        u = create_function((*fs, dim), u, use_cache=Ellipsis)
        return get_component_functions(fs, u, names, use_cache)


def create_constant(
    mesh: Mesh,
    value: float | Iterable[float] | Constant,
    try_identity: bool = False,
) -> Constant:
    """
    Typecast to `dolfinx.fem.Constant` 
    """
    if try_identity and isinstance(value, Constant) and value.ufl_domain() is mesh.ufl_domain():
        return value
    else:
        return _create_constant(value, mesh)

@singledispatch
def _create_constant(value, _):
    raise MultipleDispatchTypeError(value)

@_create_constant.register(float)
@_create_constant.register(int)
def _(value, mesh: Mesh,):
    return Constant(mesh, float(value))


@_create_constant.register(Iterable)
def _(value: Iterable[float], mesh: Mesh):
    if all(isinstance(i, (float, int)) for i in value):
        value = [float(i) for i in value]
        return Constant(mesh, value)
    else:
        raise TypeError('Expected an iterable of numbers')


def _as_function_space_tuple(
    elem: tuple[str, int] 
    | tuple[str, int, int] 
    | tuple[Mesh, str, int] 
    | tuple[Mesh, str, int, int], 
    u: Function | Expr,
) -> tuple[Mesh, str, int] | tuple[Mesh, str, int, int]:
    match elem:
        case mesh, fam, deg, dim:
            return mesh, fam, deg, dim
        case mesh, fam, deg if isinstance(mesh, Mesh):
            return mesh, fam, deg
        case fam, deg, dim:
            mesh = extract_mesh(u)
            return mesh, fam, deg, dim
        case fam, deg:
            mesh = extract_mesh(u)
            return mesh, fam, deg
        case _:
            raise TypeError