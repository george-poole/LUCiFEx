from collections.abc import Callable, Iterable
from functools import singledispatch

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
from .py_utils import MultipleDispatchTypeError, optional_lru_cache
from .fem_mutate import set_finite_element_function


def function_space(
    fs: FunctionSpace
        | tuple[Mesh, str, int]
        | tuple[Mesh, str, int]
        | tuple[Mesh, str, int, int]
        | tuple[Mesh, Iterable[tuple[str, int] | tuple[str, int, int]]],
    subspace_index: int | None = None,
    use_cache: bool = False,
) -> FunctionSpace:
    """
    Typecast to `dolfinx.fem.FunctionSpace` 
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


def finite_element_function(
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[Mesh, str, int, int] | tuple[str, int] | tuple[str, int, int],
    value: Function | Constant | Expression | Expr | Callable[[np.ndarray], np.ndarray] | float | Iterable[float | Callable[[np.ndarray], np.ndarray]],
    subspace_index: int | None = None,
    name: str | None = None,
    use_cache: bool | tuple[bool, bool] = False,
    try_identity: bool = False,
) -> Function:
    """
    Typecast to `dolfinx.fem.Function` 
    """
    if name is None:
        try:
            name = value.name
        except AttributeError:
            pass

    if isinstance(use_cache, bool):
        use_cache = (use_cache, use_cache)
            
    if isinstance(fs, FunctionSpace):
        if try_identity and isinstance(value, Function) and value.function_space == fs:
            return value
        fs = function_subspace(fs, subspace_index)
        f = Function(fs, name=name)
    else:
        fs = _as_function_space_tuple(fs, value)
        if try_identity and isinstance(value, Function) and is_same_element(value, *fs[1:], mesh=fs[0]):
            return value
        use_func_cache, use_fs_cache = use_cache
        f = _finite_element_function(use_cache=use_func_cache)(fs, name, use_fs_cache)
    set_finite_element_function(f, value)
    return f


@optional_lru_cache
def _finite_element_function(
    fs_hashable: tuple, 
    name: str | None,
    use_fs_cache: bool,
) -> Function:
    fs = _function_space(use_cache=use_fs_cache)(fs_hashable)
    return Function(fs, name=name)


@optional_lru_cache
def finite_element_subfunctions(
    u: Function,  
    collapse: bool = True,
) -> tuple[Function, ...]:
    if collapse:
        return tuple(i.collapse() for i in u.split())
    else:
        return u.split()


def finite_element_function_components(
    fs: tuple[Mesh, str, int] | tuple[str, int],
    u: Function | Expr,
    names: Iterable[str | None] | None = None,
    use_cache: bool | tuple[bool, bool] = False,
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
            u_name = f'u{id(u)}'
        names = tuple(f'{u_name}_{i}' for i in range(dim))

    if isinstance(u, Function) and is_component_space(u.function_space):
        # e.g. vector-valued u ∈ P⨯P
        # fs = function_space(fs, use_cache=use_fs_cache)
        # f = _finite_element_function_components(use_cache=use_cache)(fs, names)
        ##### f = [_finite_element_function(use_cache=False)(fs, n, use_fs_cache) for n in names]
        # u = [ui.collapse() for ui in u.split()]
        # u = finite_element_subfunctions(u)
        return tuple(
            finite_element_function(fs, i.collapse(), name=n, use_cache=use_cache) 
            for i, n in zip(u.split(), names, strict=True)
        )
    else:
        # e.g. vector-valued u ∈ BDM
        u = finite_element_function((*fs, dim), u, use_cache=use_cache)
        return finite_element_function_components(fs, u, names, use_cache)


def finite_element_constant(
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
        return _finite_element_constant(value, mesh)

@singledispatch
def _finite_element_constant(value, _):
    raise MultipleDispatchTypeError(value)

@_finite_element_constant.register(float)
@_finite_element_constant.register(int)
def _(value, mesh: Mesh,):
    return Constant(mesh, float(value))


@_finite_element_constant.register(Iterable)
def _(value: Iterable[float], mesh: Mesh):
    if all(isinstance(i, (float, int)) for i in value):
        value = [float(i) for i in value]
        return Constant(mesh, value)
    else:
        raise TypeError('Expected an iterable of numbers')


def _as_function_space_tuple(
    elem: tuple[str, int] | tuple[Mesh, str, int], 
    u: Function | Expr,
) -> tuple[Mesh, str, int]:
    match elem:
        case mesh, fam, deg:
            return elem
        case fam, deg:
            mesh = extract_mesh(u)
            return mesh, fam, deg
        case _:
            raise TypeError