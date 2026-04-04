from collections.abc import Callable, Iterable
from types import EllipsisType
from functools import singledispatch, lru_cache

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from petsc4py import PETSc
from dolfinx.fem import Function, FunctionSpace, Constant, Function
from dolfinx.fem import Expression

from ..py_utils import OverloadTypeError, StrSlice, optional_lru_cache, as_slice
from ..npy_utils import AnyNumber
from .elem_utils import is_same_element
from .expr_utils import ( 
    is_scalar,
    is_vector, 
    NonVectorError,
    ShapeError
)
from .function_space_utils import (
    extract_subspace, create_function_space,
    is_component_space, fs_from_elem,
)


def as_function(
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[Mesh, str, int, int] 
    | tuple[str, int] | tuple[str, int, int],
    value: Function | Constant | Expression | Expr
    | Callable[[np.ndarray], np.ndarray] | float 
    | Iterable[float | Callable[[np.ndarray], np.ndarray]]
    | None = None,
    subspace_index: int | None = None,
    dofs_indices: Iterable[int] | StrSlice | None = None,
    name: str | None = None,
    use_cache: bool | EllipsisType = False,
) -> Function:
    """
    If `value` is of type `Function` and already belongs to the specified function space,
    then it is returned unmutated. Otherwise a new `Function` is created.
    """
    return _as_or_create_function(
        False,
        fs,
        value,
        subspace_index,
        dofs_indices,
        name,
        use_cache,
    )


def create_function(
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[Mesh, str, int, int] 
    | tuple[str, int] | tuple[str, int, int],
    value: Function | Constant | Expression | Expr
    | Callable[[np.ndarray], np.ndarray] | float 
    | Iterable[float | Callable[[np.ndarray], np.ndarray]]
    | None = None,
    subspace_index: int | None = None,
    dofs_indices: Iterable[int] | StrSlice | None = None,
    name: str | None = None,
    use_cache: bool | EllipsisType = False,
) -> Function:
    """
    Typecast to `dolfinx.fem.Function`. 

    `use_cache = True` will try to fetch a `Function` from the cache, based on the tuple-valued `fs`, 
    `subspace_index` and `name` and then mutate that `Function`. \\
    `use_cache = Ellipsis` will try to fetch a `FunctionSpace` from the cache, but create a new `Function`. \\
    `use_
    """
    return _as_or_create_function(
        True,
        fs,
        value,
        subspace_index,
        dofs_indices,
        name,
        use_cache,
    )



def _as_or_create_function(
    create: bool,
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[Mesh, str, int, int] 
    | tuple[str, int] | tuple[str, int, int],
    value: Function | Constant | Expression | Expr
    | Callable[[np.ndarray], np.ndarray] | float 
    | Iterable[float | Callable[[np.ndarray], np.ndarray]]
    | None = None,
    subspace_index: int | None = None,
    dofs_indices: Iterable[int] | StrSlice | None = None,
    name: str | None = None,
    use_cache: bool | EllipsisType = False,
) -> Function:
    if name is None:
        try:
            name = value.name
        except AttributeError:
            pass
            
    if isinstance(fs, FunctionSpace):
        fs = extract_subspace(fs, subspace_index, collapse=True)
        if not create and isinstance(value, Function) and value.function_space == fs:
            return value
        f = Function(fs, name=name)
    else:
        if not isinstance(fs[0], Mesh):
            fs = fs_from_elem(fs, value)
        if not create and isinstance(value, Function) and is_same_element(value, *fs[1:], mesh=fs[0]):
            return value
        if use_cache is True:
            f = _create_function(fs, subspace_index, name)
        else:
            fs = create_function_space(fs, subspace_index, bool(use_cache))
            f = Function(fs, name=name)

    if value is not None:
        set_function(f, value, dofs_indices)
    
    return f


@lru_cache
def _create_function(
    fs_hashable: tuple, 
    subspace_index: int | None,
    name: str | None,
) -> Function:
    fs = create_function_space(fs_hashable, subspace_index, use_cache=True)
    return Function(fs, name=name)


@optional_lru_cache
def extract_subfunctions(
    u: Function,  
    collapse: bool = True,
) -> tuple[Function, ...]:
    if collapse:
        return tuple(i.collapse() for i in u.split())
    else:
        return u.split()


def extract_component_functions(
    fs: tuple[Mesh, str, int] | tuple[str, int],
    u: Function | Expr,
    names: Iterable[str | None] | None = None,
    create: bool = True,
    use_cache: bool | EllipsisType = False,
) -> tuple[Function, ...]:
    """
    `u(𝐱) ∈ Pₖ ⨯ Pₖ ⨯ ... -> (uˣ(𝐱), u(𝐱)ʸ, ...)`

    `u(𝐱) ∈ BDMₖ -> (uˣ(𝐱), u(𝐱)ʸ, ...)`
    """
    
    if not is_vector(u):
        raise NonVectorError(u)
    
    if not isinstance(fs[0], Mesh):
        fs = fs_from_elem(fs, u)

    dim = u.ufl_shape[0]
    
    if names is None:
        try:
            u_name = u.name
        except AttributeError:
            u_name = f'{u.__class__.__name__}{id(u)}'
        names = tuple(f'{u_name}{i}' for i in range(dim))

    if isinstance(u, Function) and is_component_space(u.function_space):
        if create:
            _function = create_function
        else:
            _function = as_function
        return tuple(
            _function(fs, i.collapse(), name=n, use_cache=use_cache) 
            for i, n in zip(u.split(), names, strict=True)
        )
    else:
        # e.g. `u(𝐱) ∈ BDM`
        u = create_function((*fs, dim), u, use_cache=Ellipsis)
        return extract_component_functions(fs, u, names, use_cache)


def extract_subdofs(
    u: Function,
) -> tuple[np.ndarray, ...]:
    fs = u.function_space
    subdofs = []
    for i in range(fs.num_sub_spaces):
        _, dofmap = fs.sub(i).collapse()
        subdofs.append(
            u.x.array[dofmap]
        )
    return tuple(subdofs)


def set_function(
    u: Function,
    value: Function 
    | Callable[[np.ndarray], np.ndarray] 
    | Expression 
    | Expr 
    | Constant 
    | AnyNumber 
    | Iterable[AnyNumber | Constant | Callable[[np.ndarray], np.ndarray]],
    dofs_indices: Iterable[int] | StrSlice | None = None,
) -> None:
    """
    Mutates `f` by either setting its DoFs array or calling its interpolation
    method. Does not mutate `value`.
    """
    if dofs_indices is None:
        set_function_interpolate(u, value)
    else:
        set_function_dofs(u, value, dofs_indices)


def set_function_dofs(
    u: Function | np.ndarray,
    value: Function 
    | Callable[[np.ndarray], np.ndarray] 
    | Expression 
    | Expr 
    | Constant 
    | AnyNumber 
    | Iterable[AnyNumber | Constant | Callable[[np.ndarray], np.ndarray]],
    dofs_indices: Iterable[int] | StrSlice | None = None,
) -> None:
    """
    Mutates `f` by calling its setting its DoFs array. Does not mutate `value`.
    """
    if isinstance(dofs_indices, StrSlice):
        dofs_indices = as_slice(dofs_indices)
    elif isinstance(dofs_indices, Iterable):
        dofs_indices = np.array(dofs_indices)

    if isinstance(u, Function):
        u = u.x.array

    return _set_function_dofs(value, u, dofs_indices)


@singledispatch
def _set_function_dofs(value, *_, **__):
    raise OverloadTypeError(value)


@_set_function_dofs.register(Function)
def _(value: Function, arr: np.ndarray, indices: np.ndarray | None):
    # assert f.function_space == value.function_space
    arr[indices] = value.x.array[indices]


@_set_function_dofs.register(Constant)
def _(value: Constant, arr: np.ndarray, indices: np.ndarray | None):
    _set_function_dofs(value.value.item(), arr, indices)


@_set_function_dofs.register(float)
@_set_function_dofs.register(int)
def _(value, arr: np.ndarray, indices: np.ndarray | None):
    arr[indices] = value

@_set_function_dofs.register(Iterable)
def _(value, arr: np.ndarray, indices: np.ndarray | None):
    arr[indices] = value[indices]


def set_function_interpolate(
    f: Function,
    value: Function | Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float | Constant | Callable[[np.ndarray], np.ndarray]],
) -> None:
    """
    Mutates `f` by calling its `interpolate` method. Does not mutate `value`.
    """
    return _set_function_interpolate(value, f)


@singledispatch
def _set_function_interpolate(u, *_, **__):
    raise OverloadTypeError(u)


@_set_function_interpolate.register(Expr)
def _(u: Expr, f: Function):
    f.interpolate(Expression(u, f.function_space.element.interpolation_points()))

@_set_function_interpolate.register(Function)
@_set_function_interpolate.register(Expression)
@_set_function_interpolate.register(Callable)
def _(value, f: Function):
    f.interpolate(value)

@_set_function_interpolate.register(Constant)
def _(value: Constant, f: Function):
    if is_scalar(value):
        return _set_function_interpolate(value.value.item(), f)
    else:
        if not f.ufl_shape == value.value.shape:
            raise ShapeError(f, value.value.shape)
        return _set_function_interpolate(value.value, f)
    

@_set_function_interpolate.register(Iterable)
def _(value: Iterable[float | Constant | Callable], f: Function):
    def _(u) -> Callable[[np.ndarray], np.ndarray]:
        if isinstance(u, (Function, Expr)):
            raise TypeError
        if isinstance(u, (int ,float)):
            return lambda x: np.full_like(x[0], u, dtype=PETSc.ScalarType)
        if isinstance(u, Constant):
            assert is_scalar(u)
            return _(u.value.item())
        if isinstance(u, Callable):
            return u
        raise OverloadTypeError(u)
    f.interpolate(lambda x: np.vstack([_(i)(x) for i in value]))


@_set_function_interpolate.register(float)
@_set_function_interpolate.register(int)
def _(value, f: Function):
    f.interpolate(lambda x: np.full_like(x[0], value, dtype=PETSc.ScalarType))