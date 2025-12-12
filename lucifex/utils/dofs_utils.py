from typing import TypeAlias
from types import EllipsisType
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

from .fem_utils import (
    create_fem_function, 
    get_component_fem_functions, 
    set_fem_function_dofs,
)
from .ufl_utils import is_scalar, is_vector, ScalarVectorError
from .py_utils import StrEnum


SpatialExpression = Callable[[np.ndarray], np.ndarray]
"""
Function of coordinates `x = (x₀, x₁, x₂)` returning an expression `f(x)`
such that `f(x) = 0 ` defines the boundary.

e.g. `lambda x: x[1] - Ly` if a boundary is defined by `y = Ly`
"""
SpatialMarker = Callable[[np.ndarray], bool]
"""
Function of coordinates `x = (x₀, x₁, x₂)` returning `True` or `False`

e.g. `lambda x: np.isclose(x[1], Ly)` if a boundary is defined by `y = Ly`
"""

# TODO int and str markers from gmsh meshtags
SpatialMarkerAlias: TypeAlias = SpatialExpression | Iterable[SpatialExpression | SpatialMarker]
"""
Alias types to `SpatialMarker`.
"""
SubspaceIndex: TypeAlias = int | None 


class DofsMethodType(StrEnum):
    GEOMETRICAL = 'geometrical'
    TOPOLOGICAL = 'topological'


def dofs_indices(
    fs: FunctionSpace,
    dofs_marker: SpatialMarker | SpatialMarkerAlias,
    subspace_index: int | None = None,
    method: DofsMethodType = DofsMethodType.TOPOLOGICAL,
) -> np.ndarray | list[np.ndarray]:
    
    method = DofsMethodType(method)
    _dofs_marker = as_spatial_marker(dofs_marker)

    if method == DofsMethodType.GEOMETRICAL:
        if subspace_index is None:
            return locate_dofs_geometrical(fs, _dofs_marker)
        else:
            function_subspace, _ = fs.sub(subspace_index).collapse()
            return locate_dofs_geometrical(
                [fs.sub(subspace_index), function_subspace],
                _dofs_marker,
            )
        
    if method == DofsMethodType.TOPOLOGICAL:
        tdim = fs.mesh.topology.dim
        edim = tdim - 1
        facets = locate_entities(
            fs.mesh, edim, _dofs_marker
        )
        if subspace_index is None:
            return locate_dofs_topological(fs, edim, facets)
        else:
            function_subspace, _ = fs.sub(subspace_index).collapse()
            dofs = locate_dofs_topological(
                [fs.sub(subspace_index), function_subspace],
                edim,
                facets,
            )
            assert len(dofs) == 2
            return dofs
        
    raise ValueError(f'{method} not recognised')

        
def dofs(
    u: Function | Expr,
    fs: FunctionSpace | tuple[Mesh, str, int] | tuple[str, int] | None = None,
    l2_norm: bool = False,
    use_cache: bool | EllipsisType | tuple = False,
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
        u = create_fem_function(fs, u, try_identity=try_identity, use_cache=use_cache)
        return u.x.array[:]
    elif l2_norm and is_vector(u):
        if not isinstance(use_cache, tuple):
            use_cache = (use_cache, use_cache)
        use_scalars_cache, use_vector_cache= use_cache
        component_dofs = np.stack(
            [
                dofs(i, fs, use_cache=use_scalars_cache, try_identity=False) 
                for i in get_component_fem_functions(fs, u, use_cache=use_vector_cache)
            ], 
            axis=1,
        )
        return np.linalg.norm(component_dofs, axis=1, ord=2)
    else:
        raise ScalarVectorError(u)
    

def as_spatial_marker(
    m: SpatialMarker | SpatialMarkerAlias
) -> SpatialMarker:
    """
    Converts a function of coordinates `x = (x₀, x₁, x₂)` returning expression `f(x)`, 
    such that `f(x) = 0`, defines the boundary to a function returning `True` 
    if `x` is on the boundary and `False` otherwise.
    """
    
    def _as_marker(m: SpatialMarker | SpatialMarkerAlias) -> SpatialMarker:
        x_test = np.array([0.0, 0.0, 0.0])
        if isinstance(m(x_test), (bool, np.bool_)):
            return m
        else:
            return lambda x: np.isclose(m(x), 0.0)

    if not isinstance(m, Iterable):
        return _as_marker(m)
    else:
        return lambda x: np.any([_as_marker(mi)(x) for mi in m], axis=0)
    

def limits_corrector(
    lower: float | Constant | None = None,
    upper: float | Constant | None = None,
    conservation: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[[np.ndarray], None]:
    """
    NOTE intended for use on DoFs that are pointwise evaluations (e.g. Lagrange elements)
    """
    def _(u: np.ndarray) -> None:
        if conservation:
            mass_pre = conservation(np.copy(u))

        if lower is not None:
            if isinstance(lower, Constant):
                _lower = lower.value
            else:
                _lower = lower
            u[u < _lower] = _lower
        if upper is not None:
            if isinstance(upper, Constant):
                _upper = upper.value
            else:
                _upper = upper
            u[u > _upper] = _upper
        
        if conservation:
            mass_post = conservation(np.copy(u))
            mass_len = len([i for i in u if i not in (lower, upper)])
            u[:] += (mass_post - mass_pre) / mass_len
    
    return _


def dirichlet_corrector(
    fs: FunctionSpace,
    *markers_values: tuple[SpatialMarkerAlias, float | Constant | Function | Expr] 
    | tuple[SpatialMarkerAlias, float | Constant | Function | Expr, int]
) -> Callable[[np.ndarray], None]:

    markers, values, subspace_indices = [], [], []
    for m_v_si in markers_values:
        if len(m_v_si) == 2:
            m, v, si = *m_v_si, None
        elif len(m_v_si) == 3:
            m, v, si = m_v_si
        else:
            raise ValueError
        markers.append(m)
        values.append(v)
        subspace_indices.append(si)

    dofs: list[np.ndarray] = []
    for m, v, i in zip(markers, values, subspace_indices, strict=True):
        d = dofs_indices(fs, m, i)
        if not isinstance(d, np.ndarray):
            d = d[0]
        dofs.append[d]

    def _(u: np.ndarray) -> None:
        for d, v in zip(d, values):
            set_fem_function_dofs(u, v, d)

    return _
