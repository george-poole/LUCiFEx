from collections.abc import Iterable
from typing import overload

from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, FunctionSpace, Function, VectorFunctionSpace
from ufl import FiniteElement, MixedElement, VectorElement

from ..py_utils import optional_lru_cache
from .expr_utils import (
    extract_mesh,     
    is_scalar,
)


def is_mixed_space(
    fs: FunctionSpace,
    strict: bool = False,
) -> bool:
    """e.g. Returns `True` if the function space is mixed.

    `Pₖ × ... × Pₖ` -> `True` \\
    `BDMₖ x DPₖ₋₁` -> `True`
    """
    if not strict:
        return fs.num_sub_spaces > 0
    else:
        return is_mixed_space(fs, strict=False) and not is_component_space(fs)


def is_component_space(fs: FunctionSpace) -> bool:
    """Returns `True` if the function space is a mixed space 
    in which all subspaces are identical and scalar-valued.
    
    `Pₖ × ... × Pₖ` -> `True` \\
    `BDMₖ` -> `False`
    """
    if not is_mixed_space(fs, strict=False):
        return False
    else:
        subspaces = extract_subspaces(fs)
        sub0 = subspaces[0]
        if not is_scalar(sub0):
            return False
        return all([sub.ufl_element() == sub0.ufl_element() for sub in subspaces])
    

def extract_subspace(
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


def extract_subspaces(
    fs: FunctionSpace,
    collapse: bool = True,
) -> tuple[FunctionSpace, ...]:
    subspaces = []
    n_sub = fs.num_sub_spaces
    for n in range(n_sub):
        subspaces.append(extract_subspace(fs, n, collapse))
    return tuple(subspaces)


@overload
def create_function_space(
    fs: FunctionSpace,
    subspace_index: int | None = None,
) -> FunctionSpace:
    ...


@overload
def create_function_space(
    fs: tuple[Mesh, str, int]
    | tuple[Mesh, str, int]
    | tuple[Mesh, str, int, int]
    | tuple[Mesh, Iterable[tuple[str, int] | tuple[str, int, int]]],
    subspace_index: int | None = None,
    use_cache: bool = False,
) -> FunctionSpace:
    ...


def create_function_space(
    fs,
    subspace_index=None,
    use_cache=False,
) -> FunctionSpace:
    """
    Typecast to `dolfinx.fem.FunctionSpace`.
    """
    if isinstance(fs, FunctionSpace):
        return extract_subspace(fs, subspace_index)
    else:
        return _create_function_space(use_cache=use_cache)(fs, subspace_index)
    

@optional_lru_cache
def _create_function_space(
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

    return extract_subspace(fs, subspace_index)


def fs_from_elem(
    elem: tuple[str, int] | tuple[str, int, int],
    u: Function | Expr,
) -> tuple[Mesh, str, int] | tuple[Mesh, str, int, int]:
    mesh = extract_mesh(u)
    return mesh, *elem


def is_equivalent_space(
    fs: FunctionSpace,
    other: FunctionSpace,
    strict: bool = False,
) -> bool:
    if strict:
        return fs is other
    if fs is other:
        return True
    if not fs.mesh is other.mesh:
        return False
    return fs.element == other.element
    
