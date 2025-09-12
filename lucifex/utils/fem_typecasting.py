from collections.abc import Callable, Iterable
from functools import singledispatch, lru_cache

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, FunctionSpace, Constant, Function, VectorFunctionSpace
from dolfinx.fem import Expression
from ufl import FiniteElement, MixedElement, VectorElement

from .fem_utils import is_same_element, extract_mesh, is_vector, is_component_space, VectorError
from .py_utils import MultipleDispatchTypeError, optional_lru_cache
from .fem_mutation import interpolate_fem_function


def fem_function_space(
    fs: FunctionSpace
        | tuple[Mesh, str, int]
        | tuple[Mesh, str, int]
        | tuple[Mesh, str, int, int]
        | tuple[Mesh, Iterable[tuple[str, int] | tuple[str, int, int]]],
    subspace_index: int | None = None,
) -> FunctionSpace:
    
    match fs:
        case fs if isinstance(fs, FunctionSpace):
            function_space = fs

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
            function_space = FunctionSpace(mesh, MixedElement(*mixed_elements))

        case mesh, fam, deg:
            function_space = FunctionSpace(mesh, (fam, deg))
        
        case mesh, fam, deg, dim:
            function_space = VectorFunctionSpace(mesh, (fam, deg), dim)
        
        case _:
            raise ValueError(f'{fs}')
        
    if subspace_index is not None:
        function_space, _ = function_space.sub(subspace_index).collapse()

    return function_space


def fs_from_elem(
    elem: tuple[str, int] | tuple[Mesh, str, int], 
    u: Function | Expr,
) -> tuple[Mesh, str, int]:
    if isinstance(elem, tuple) and not isinstance(elem[0], Mesh):
        if isinstance(u, Function):
            mesh = u.function_space.mesh
        elif isinstance(u, (Expression, Expr)):
            mesh = extract_mesh(u)
        else:
            raise ValueError('Cannot deduce mesh from `u`. Provide function space as `tuple[Mesh, str, int]`')
        fs = (mesh, *elem)
    else:
        fs = elem
    return fs


@optional_lru_cache
def _create_fem_function(
    fs: FunctionSpace | tuple[Mesh, str, int], 
    subspace_index: int | None = None,
    name: str | None = None,
) -> Function:
    fs = fem_function_space(fs, subspace_index)
    return Function(fs, name=name)


def fem_function(
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[Mesh, str, int, int] | tuple[str, int] | tuple[str, int, int],
    u: Function | Constant | Expression | Expr | Callable[[np.ndarray], np.ndarray] | Iterable[float | Callable[[np.ndarray], np.ndarray]],
    subspace_index: int | None = None,
    name: str | None = None,
    use_cache: bool = False,
    recycle: bool = False,
) -> Function:
            
    fs = fs_from_elem(fs, u)

    if recycle and isinstance(u, Function):
        if isinstance(fs, FunctionSpace) and u.function_space == fs:
            return u
        if isinstance(fs, tuple):
            if isinstance(fs[0], Mesh):
                mesh = fs[0]
                _fs = fs[1:]
            else:
                mesh = None
                _fs = fs
            if is_same_element(u, *_fs, mesh=mesh):
                return u
        
    if name is None:
        try:
            name = u.name
        except AttributeError:
            pass

    f = _create_fem_function(use_cache=use_cache)(fs, subspace_index, name)
    interpolate_fem_function(f, u)

    return f


@optional_lru_cache
def _create_fem_function_components(
    fs: FunctionSpace | tuple[Mesh, str, int], 
    names: tuple[str | None, ...],
) -> tuple[Function, ...]:
    f = [_create_fem_function(use_cache=False)(fs, name=n) for n in names]
    return tuple(f)


def fem_function_components(
    fs: tuple[Mesh, str, int] | tuple[str, int],
    u: Function | Expr,
    names: Iterable[str | None] | None = None,
    use_cache: bool = False,
) -> tuple[Function, ...]:
    
    if not is_vector(u):
        raise VectorError(u)
    
    fs = fs_from_elem(fs, u)
    dim = u.ufl_shape[0]
    
    if names is None:
        try:
            names = tuple(f'{u.name}_{i}' for i in range(dim))
        except AttributeError:
            names = tuple([None] * dim)

    if isinstance(u, Function) and is_component_space(u.function_space):
        # e.g. vector-valued u ∈ P⨯P
        f = _create_fem_function_components(use_cache=use_cache)(fs, names)
        u = [ui.collapse() for ui in u.split()]
        for fi, ui in zip(f, u, strict=True):
            interpolate_fem_function(fi, ui)
        return f
    else:
        # e.g. vector-valued u ∈ BDM
        u = fem_function((*fs, dim), u, use_cache=use_cache)
        return fem_function_components(fs, u, names, use_cache)


def fem_constant(
    mesh: Mesh,
    value: float | Iterable[float] | Constant,
) -> Constant:
    if isinstance(value, Constant):
        return value
    else:
        return _fem_constant(value, mesh)

@singledispatch
def _fem_constant(value, _):
    raise MultipleDispatchTypeError(value)

@_fem_constant.register(float)
@_fem_constant.register(int)
def _(value, mesh: Mesh,):
    return Constant(mesh, float(value))


@_fem_constant.register(Iterable)
def _(value: Iterable[float], mesh: Mesh):
    if all(isinstance(i, (float, int)) for i in value):
        return Constant(mesh, value)
    else:
        raise TypeError('Expected an iterable of numbers')