from typing import TypeAlias, TypeVar
from collections.abc import Callable, Iterable

import numpy as np
from dolfinx.mesh import locate_entities, Mesh
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from ufl.core.expr import Expr

from .py_utils import StrEnum
from .fem_typecasting import fem_function, fem_function_components
from .fem_utils import is_scalar, is_vector, ScalarVectorError
from .fem_mutation import set_fem_function


SpatialExpression = Callable[[np.ndarray], np.ndarray]
"""
Function of coordinates `x = (x₀, x₁, x₂)` returning expression `f(x)`

e.g. `lambda x: x[1] - Ly` if a boundary is defined by `y = Ly`
"""

SpatialMarker = Callable[[np.ndarray], bool]
"""
Function of coordinates `x = (x₀, x₁, x₂)` returning `True` or `False`

e.g. `lambda x: np.isclose(x[1], Ly)` if a boundary is defined by `y = Ly`
"""

SpatialMarkerOrExpression: TypeAlias = SpatialMarker | SpatialExpression
"""
Function of coordinates `x = (x₀, x₁, x₂)` either returning expression `f(x)` 
such that `f(x) = 0` defines the boundary, or `True` if `x` is on the boundary and `False` otherwise.
"""

# TODO int and str markers from mesh tags (especially irregular gmsh meshes)?
SpatialMarkerTypes: TypeAlias = SpatialMarkerOrExpression | Iterable[SpatialMarkerOrExpression]

SubspaceIndex: TypeAlias = int | None 


class DofsMethodType(StrEnum):
    GEOMETRICAL = 'geometrical'
    TOPOLOGICAL = 'topological'


def dofs_indices(
    function_space: FunctionSpace,
    dofs_marker: SpatialMarkerTypes,
    subspace_index: int | None = None,
    method: DofsMethodType = DofsMethodType.TOPOLOGICAL,
) -> np.ndarray | list[np.ndarray]:
    
    method = DofsMethodType(method)
    _dofs_marker = as_spatial_marker(dofs_marker)

    if method == DofsMethodType.GEOMETRICAL:
        if subspace_index is None:
            return locate_dofs_geometrical(function_space, _dofs_marker)
        else:
            function_subspace, _ = function_space.sub(subspace_index).collapse()
            return locate_dofs_geometrical(
                [function_space.sub(subspace_index), function_subspace],
                _dofs_marker,
            )
        
    if method == DofsMethodType.TOPOLOGICAL:
        tdim = function_space.mesh.topology.dim
        edim = tdim - 1
        facets = locate_entities(
            function_space.mesh, edim, _dofs_marker
        )
        if subspace_index is None:
            return locate_dofs_topological(function_space, edim, facets)
        else:
            function_subspace, _ = function_space.sub(subspace_index).collapse()
            dofs = locate_dofs_topological(
                [function_space.sub(subspace_index), function_subspace],
                edim,
                facets,
            )
            assert len(dofs) == 2
            return dofs
        
    raise ValueError(f'{method} not recognised')


def as_spatial_marker(
    m: SpatialMarkerOrExpression | Iterable[SpatialMarkerOrExpression]
) -> SpatialMarker:
    """
    Converts a function of coordinates `x = (x₀, x₁, x₂)` returning expression `f(x)` 
    such that `f(x) = 0` defines the boundary to a function returning `True` 
    if `x` is on the boundary and `False` otherwise.
    """
    
    def _as_marker(m: SpatialMarkerOrExpression) -> SpatialMarker:
        x_test = np.array([0.0, 0.0, 0.0])
        if isinstance(m(x_test), (bool, np.bool_)):
            return m
        else:
            return lambda x: np.isclose(m(x), 0.0)

    if not isinstance(m, Iterable):
        return _as_marker(m)
    else:
        return lambda x: np.any([_as_marker(mi)(x) for mi in m], axis=0)
    
    
# @optional_lru_cache
def dofs(
    u: Function | Expr,
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[str, int] | None = None,
    l2_norm: bool = True,
    use_cache: bool = False,
    try_identity: bool = False,
) -> np.ndarray:
    """
    scalar `u(𝐱) = Σᵢ Uᵢϕᵢ(𝐱)` returns `{Uᵢ}`

    vector `𝐮(𝐱) = Σᵢ Uᵢ𝛟ᵢ(𝐱)` and `l2_norm=False` returns `{Uᵢ}`
    
    vector `𝐮(𝐱) = Σᵢ (Uˣᵢ, Uʸᵢ, Uᶻᵢ)ϕᵢ(𝐱)` and `l2_norm=True` returns `{(Uˣᵢ² + Uʸᵢ² + Uᶻᵢ²)¹ᐟ²}`
    """
    
    if fs is None:
        assert isinstance(u, Function)
        fs = u.function_space
    
    if is_scalar(u) or (not l2_norm and is_vector(u)):
        u = fem_function(fs, u, use_cache=use_cache, try_identity=try_identity)
        return u.x.array[:]  # TODO or .vector[:] ?
    elif l2_norm and is_vector(u):
        scalars = fem_function_components(fs, u, use_cache=True)
        scalar_dofs = np.stack([dofs(i, fs, use_cache=False, try_identity=False) for i in scalars], axis=1)
        return np.linalg.norm(scalar_dofs, axis=1, ord=2)
    else:
        raise ScalarVectorError(u)


T = TypeVar('T')
def dofs_transformation(
    func: Callable[[np.ndarray], T] | Iterable[Callable[[np.ndarray], T]],
    u: Function | Expr,
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[str, int] | None = None,
    l2_norm: bool = True,
    use_cache: bool = False,
) -> T | tuple[T, ...]:
    if isinstance(func, Iterable):
        return tuple(dofs_transformation(f, u, fs, l2_norm, use_cache) for f in func)
    return func(dofs(u, fs, l2_norm, use_cache))


def extremum(
    u: Function | Expr,
    fs: tuple[str, int] = ('P', 1),
) -> tuple[float, float]:
    return dofs_transformation((np.min, np.max), u, fs, use_cache=True)


def minimum(
    u: Function | Expr,
    fs: tuple[str, int] = ('P', 1),
) -> float:
    return dofs_transformation(np.min, u, fs, use_cache=True)


def maximum(
    u: Function | Expr,
    fs: tuple[str, int] = ('P', 1),
) -> float:
    return dofs_transformation(np.max, u, fs, use_cache=True)


def as_dofs_setter(
    setter: Callable[[Function], None] 
    | Iterable[tuple[SpatialMarkerTypes, float | Constant] | tuple[SpatialMarkerTypes, float | Constant, int]]
    | None,
) -> Callable[[Function], None]:
    
    if isinstance(setter, Callable):
        return setter
    
    if setter is None:
        return as_dofs_setter([])

    markers, values, subspace_indices = [], [], []
    for sttr in setter:
        if len(sttr) == 2:
            m, v, si = *sttr, None
        elif len(sttr) == 3:
            m, v, si = sttr
        else:
            raise ValueError
        markers.append(as_spatial_marker(m))
        values.append(v)
        subspace_indices.append(si)
    
    def _corrector(f: Function) -> None:
        for m, v, i in zip(markers, values, subspace_indices, strict=True):
            dofs = dofs_indices(f.function_space, m, i)
            if not isinstance(dofs, np.ndarray):
                dofs = dofs[0]
            set_fem_function(f, v, dofs)

    return _corrector


def dofs_limits_corrector(
    u: Function,
    limits: tuple[float | None, float | None] | bool | None,
) -> None:
    """
    Enforces `u ∈ [umin, umax]` 

    NOTE assumes DoFs are pointwise evaluations (e.g. Lagrange elements)
    """
    if limits is None:
        limits = (None, None)
    umin, umax = limits
    if umin is not None:
        u.x.array[u.x.array < umin] = umin
    if umax is not None:
        u.x.array[u.x.array > umax] = umax