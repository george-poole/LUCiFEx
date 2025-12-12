from collections.abc import Callable, Iterable
from typing import overload
from types import EllipsisType
from functools import singledispatch, lru_cache

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from petsc4py import PETSc
from dolfinx.fem import Function, FunctionSpace, Constant, Function, VectorFunctionSpace
from dolfinx.fem import Expression
from ufl import FiniteElement, MixedElement, VectorElement

from .ufl_utils import (
    is_same_element,
    extract_mesh,     
    is_scalar,
    is_vector, 
    VectorError,
    ShapeError
)
from .py_utils import MultipleDispatchTypeError, StrSlice, optional_lru_cache, as_slice


def is_mixed_space(fs: FunctionSpace) -> bool:
    """e.g. Returns `True` if the function space is mixed.

    `Pₖ × ... × Pₖ` -> `True` \\
    `BDMₖ x DPₖ₋₁` -> `True`
    """
    return fs.num_sub_spaces > 0


def is_component_space(fs: FunctionSpace) -> bool:
    """Returns `True` if the function space is a mixed space 
    in which all subspaces are identical and scalar-valued.
    
    `Pₖ × ... × Pₖ` -> `True` \\
    `BDMₖ` -> `False`
    """
    if not is_mixed_space(fs):
        return False
    else:
        subspaces = get_fem_subspaces(fs)
        sub0 = subspaces[0]
        if not is_scalar(sub0):
            return False
        return all([sub.ufl_element() == sub0.ufl_element() for sub in subspaces])
    

def get_fem_subspace(
    fs: FunctionSpace,
    subspace_index: int | None = None,
    collapse: bool = True,
) -> FunctionSpace:
    if subspace_index is None:
        return fs
    fs_sub = fs.sub(subspace_index)
    if collapse:
        return fs_sub.collapse()[0]
    else:
        return fs_sub


def get_fem_subspaces(
    fs: FunctionSpace,
    collapse: bool = True,
) -> tuple[FunctionSpace, ...]:
    subspaces = []
    n_sub = fs.num_sub_spaces
    for n in range(n_sub):
        subspaces.append(get_fem_subspace(fs, n, collapse))
    return tuple(subspaces)


@overload
def create_fem_space(
    fs: FunctionSpace,
    subspace_index: int | None = None,
) -> FunctionSpace:
    ...


@overload
def create_fem_space(
    fs: tuple[Mesh, str, int]
    | tuple[Mesh, str, int]
    | tuple[Mesh, str, int, int]
    | tuple[Mesh, Iterable[tuple[str, int] | tuple[str, int, int]]],
    subspace_index: int | None = None,
    use_cache: bool = False,
) -> FunctionSpace:
    ...


def create_fem_space(
    fs,
    subspace_index=None,
    use_cache=False,
) -> FunctionSpace:
    """
    Typecast to `dolfinx.fem.FunctionSpace`.
    """
    if isinstance(fs, FunctionSpace):
        return get_fem_subspace(fs, subspace_index)
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

    return get_fem_subspace(fs, subspace_index)


def create_fem_function(
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
        fs = get_fem_subspace(fs, subspace_index)
        f = Function(fs, name=name)
    else:
        fs = _as_function_space_tuple(fs, value)
        if try_identity and isinstance(value, Function) and is_same_element(value, *fs[1:], mesh=fs[0]):
            return value
        if use_cache is True:
            f = _create_function(fs, subspace_index, name)
        else:
            fs = create_fem_space(fs, subspace_index, bool(use_cache))
            f = Function(fs, name=name)

    set_fem_function(f, value, dofs_indices)
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


def get_component_fem_functions(
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
            create_fem_function(fs, i.collapse(), name=n, use_cache=use_cache) 
            for i, n in zip(u.split(), names, strict=True)
        )
    else:
        # e.g. `u(𝐱) ∈ BDM`
        u = create_fem_function((*fs, dim), u, use_cache=Ellipsis)
        return get_component_fem_functions(fs, u, names, use_cache)


def create_fem_constant(
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


def set_fem_function(
    f: Function,
    value: Function 
    | Callable[[np.ndarray], np.ndarray] 
    | Expression 
    | Expr 
    | Constant 
    | float 
    | Iterable[float | Constant | Callable[[np.ndarray], np.ndarray]],
    dofs_indices: Iterable[int] | StrSlice | None = None,
) -> None:
    """
    Mutates `f` by either setting its DoFs array or calling its interpolation
    method. Does not mutate `value`.
    """
    if dofs_indices is None:
        set_fem_function_interpolate(f, value)
    else:
        set_fem_function_dofs(f, value, dofs_indices)


def set_fem_function_dofs(
    f: Function | np.ndarray,
    value: Function 
    | Callable[[np.ndarray], np.ndarray] 
    | Expression 
    | Expr 
    | Constant 
    | float 
    | Iterable[float | Constant | Callable[[np.ndarray], np.ndarray]],
    dofs_indices: Iterable[int] | StrSlice | None = None,
) -> None:
    """
    Mutates `f` by calling its setting its DoFs array. Does not mutate `value`.
    """
    if isinstance(dofs_indices, StrSlice):
        dofs_indices = as_slice(dofs_indices)
    elif isinstance(dofs_indices, Iterable):
        dofs_indices = np.array(dofs_indices)

    if isinstance(f, Function):
        f = f.x.array

    return _set_fem_function_dofs(value, f, dofs_indices)


@singledispatch
def _set_fem_function_dofs(value, *_, **__):
    raise MultipleDispatchTypeError(value)


@_set_fem_function_dofs.register(Function)
def _(value: Function, arr: np.ndarray, indices: np.ndarray | None):
    # assert f.function_space == value.function_space
    arr[indices] = value.x.array[indices]


@_set_fem_function_dofs.register(Constant)
def _(value: Constant, arr: np.ndarray, indices: np.ndarray | None):
    _set_fem_function_dofs(value.value.item(), arr, indices)


@_set_fem_function_dofs.register(float)
@_set_fem_function_dofs.register(int)
def _(value, arr: np.ndarray, indices: np.ndarray | None):
    arr[indices] = value

@_set_fem_function_dofs.register(Iterable)
def _(value, arr: np.ndarray, indices: np.ndarray | None):
    arr[indices] = value[indices]


def set_fem_function_interpolate(
    f: Function,
    value: Function | Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float | Constant | Callable[[np.ndarray], np.ndarray]],
) -> None:
    """
    Mutates `f` by calling its `interpolate` method. Does not mutate `value`.
    """
    return _set_fem_function_interpolate(value, f)


@singledispatch
def _set_fem_function_interpolate(u, *_, **__):
    raise MultipleDispatchTypeError(u)


@_set_fem_function_interpolate.register(Expr)
def _(u: Expr, f: Function):
    f.interpolate(Expression(u, f.function_space.element.interpolation_points()))

@_set_fem_function_interpolate.register(Function)
@_set_fem_function_interpolate.register(Expression)
@_set_fem_function_interpolate.register(Callable)
def _(value, f: Function):
    f.interpolate(value)

@_set_fem_function_interpolate.register(Constant)
def _(value: Constant, f: Function):
    if is_scalar(value):
        return _set_fem_function_interpolate(value.value.item(), f)
    else:
        if not f.ufl_shape == value.value.shape:
            raise ShapeError(f, value.value.shape)
        return _set_fem_function_interpolate(value.value, f)
    

@_set_fem_function_interpolate.register(Iterable)
def _(value: Iterable[float | Constant | Callable], f: Function):
    def _inner(u) -> Callable[[np.ndarray], np.ndarray]:
        if isinstance(u, (int ,float)):
            return lambda x: np.full_like(x[0], u, dtype=PETSc.ScalarType)
        if isinstance(u, Constant):
            assert is_scalar(u)
            return _(u.value.item())
        if isinstance(u, Callable):
            return u
        raise MultipleDispatchTypeError(u)
    f.interpolate(lambda x: np.vstack([_inner(i)(x) for i in value]))


@_set_fem_function_interpolate.register(float)
@_set_fem_function_interpolate.register(int)
def _(value, f: Function):
    f.interpolate(lambda x: np.full_like(x[0], value, dtype=PETSc.ScalarType))


def set_fem_constant(
    c: Constant,
    value: Constant | float | np.ndarray | Iterable[float],
) -> None:
    """
    Mutates `c` by setting its value array. Does not mutate `value`.
    """
    return _set_finite_element_constant(value, c)


@singledispatch
def _set_finite_element_constant(value, *_, **__):
    raise MultipleDispatchTypeError(value)


@_set_finite_element_constant.register(Constant)
def _(value: Constant, const: Constant):
    const.value = value.value.copy()


@_set_finite_element_constant.register(float)
@_set_finite_element_constant.register(int)
def _(value, const: Constant):
    const.value = value

@_set_finite_element_constant.register(np.ndarray)
def _(value: np.ndarray, const: Constant):
    if not const.value.shape == value.shape:
        raise ShapeError(const, value.shape)
    const.value = value

@_set_finite_element_constant.register(Iterable)
def _(value, const: Constant):
    return _set_finite_element_constant(np.array(value), const)
