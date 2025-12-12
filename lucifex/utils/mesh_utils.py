from typing import Iterable, Literal, ParamSpec, TypeVar, Callable, overload
from collections.abc import Iterable
from functools import wraps


import numpy as np
from ufl import Measure
from ufl.core.expr import Expr
from dolfinx.fem import assemble_scalar, form
from dolfinx.cpp.mesh import entities_to_geometry
from dolfinx.mesh import CellType as DolfinxCellType
from dolfinx.mesh import DiagonalType as DolfinxDiagonalType
from dolfinx.mesh import Mesh, locate_entities, meshtags
from ufl.geometry import (GeometricCellQuantity, Circumradius, 
                          CellDiameter, MaxCellEdgeLength, MinCellEdgeLength)

from .py_utils import optional_lru_cache, StrEnum
from .fem_utils import extract_mesh
from .dofs_utils import (
    dofs,
    as_spatial_marker,
    SpatialMarkerAlias,
)


class CellType(StrEnum): 
    """Enumeration class containing implemented cell types"""

    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    TETRAHEDRON = "tetrahedron"
    HEXAHEDRON = "hexahedron"
    
    @property
    def cpp_type(self):
        return getattr(DolfinxCellType, self.value)


class DiagonalType(StrEnum): 
    """Enumeration class containing implemented cell diagonal types"""

    LEFT = "left"
    RIGHT = "right"
    LEFT_RIGHT = "left_right"
    RIGHT_LEFT = "right_left"
    CROSSED = "crossed"
    
    @property
    def cpp_type(self):
        return getattr(DolfinxDiagonalType, self.value)
    

class BoundaryType(StrEnum): 
    """Enumeration class containing implemented boundary condition types"""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    ESSENTIAL = "essential"
    NATURAL = "natural"
    PERIODIC = "periodic"
    ANTIPERIODIC = "antiperiodic"
    WEAK_DIRICHLET = "weak_dirichlet"


def create_tagged_measure(
    measure: Literal['dx', 'ds', 'dS'],
    mesh: Mesh,
    markers: Iterable[SpatialMarkerAlias] = (),
    tags: Iterable[int] | None = None,
    tag_unmarked: int | None = None,
    **measure_kwargs,
) -> Measure:
    """
    If `tags` and `tag_unmarked` are unspecified, creates 
    a measure `dx` such that `dx(i)` is the measure for 
    the `i`th subdomain, where `0 ≤ i ≤ n - 1` where and `n` is 
    the total number of specified subdomains. `dx(n)` is the 
    measure for the complementary subdomain to the union of 
    all specified subdomains.
    """
    if not markers:
        return Measure(measure, domain=mesh, metadata=measure_kwargs)
    
    if tags is None:
        tags = list(range(len(markers)))
    if tag_unmarked is None:
        tag_unmarked = max(tags) + 1
    assert tag_unmarked not in tags

    gdim = mesh.topology.dim
    fdim = gdim - 1
    mesh.topology.create_entities(fdim)
    facet_index_map = mesh.topology.index_map(fdim)
    num_facets = facet_index_map.size_local + facet_index_map.num_ghosts
    facet_tags = np.full(num_facets, tag_unmarked, dtype=np.intc)

    for t, m in zip(tags, markers, strict=True):
        m = as_spatial_marker(m)
        facet_indices = locate_entities(mesh, fdim, m)
        facet_tags[facet_indices] = t

    mesh_tags = meshtags(mesh, fdim, np.arange(num_facets), facet_tags)
    return Measure(measure, domain=mesh, subdomain_data=mesh_tags, metadata=measure_kwargs)    

    
P = ParamSpec('P')
R = TypeVar('R') # Expr | tuple[Expr, ...]
def mesh_integral(
    func: Callable[P, R]
):
    """
    Decorator for functions returning integrand expressions.
    """
    
    @overload
    def _(
        *args: P.args, 
        **kwargs: P.kwargs,
    ) -> R:
        ...
    
    @overload
    def _(
        measure: Literal['dx', 'ds', 'dS'] | Measure, 
        *markers: SpatialMarkerAlias,
        facet_side: Literal['+', '-'] | None = None,
        **measure_kwargs,
    ) -> Callable[P, float | np.ndarray]:
        ...


    def _overload(
        measure: Literal['dx', 'ds', 'dS'] | Measure, 
        *markers: SpatialMarkerAlias,
        facet_side: Literal['+', '-'] | None = None,
        **measure_kwargs,
    ):
        """
        If more than one marker provided, returned array will have shape
        `(n_marker, )` or `(n_marker, n_expr)` for a size `n_expr` of the 
        tuple of expressions returned by the integrand function.

            
            """
        if facet_side is not None:
            _facet_side = lambda expr: expr(facet_side)
        else:
            _facet_side = lambda expr: expr

        def _inner(*a, **k):
            integrand = func(*a, **k)
            if isinstance(integrand, Expr):
                integrand = _facet_side(integrand)
            else:
                integrand = tuple(_facet_side(i) for i in integrand)

            if isinstance(measure, Measure):
                dx = measure
            else:
                if isinstance(integrand, Expr):
                    mesh = extract_mesh(integrand)
                else:
                    mesh = extract_mesh(integrand[0])
                dx_tagged = create_tagged_measure(measure, mesh, markers, **measure_kwargs)

            n_subdomains = len(markers)

            if n_subdomains == 0:
                dx = dx_tagged
            elif n_subdomains == 1:
                dx = dx_tagged(0)
            else:
                dx = [dx_tagged(i) for i in range(n_subdomains)]
            
            if isinstance(dx, Measure):
                if isinstance(integrand, tuple):
                    return np.array([assemble_scalar(form(i * dx)) for i in integrand])
                else:
                    return assemble_scalar(form(integrand * dx))
            else:
                if isinstance(integrand, tuple):
                    return np.array([[assemble_scalar(form(i * _dx)) for i in integrand] for _dx in dx])
                else:
                    return np.array([assemble_scalar(form(integrand * _dx)) for _dx in dx])
                
        return _inner

    @wraps(func)
    def _(*args, **kwargs):
        if isinstance(args[0], (Measure, str)):
            return _overload(*args, **kwargs)
        else:
            return func(*args, **kwargs)
        
    return _



def mesh_vertices(
    mesh: Mesh,
    dim: int = None,
) -> list[tuple[float, ...]]:
    """`[(x₀, y₀, z₀), (x₁, y₁, z₁), (x₂, y₂, z₂), ...]`"""
    if dim is None:
        dim = mesh.geometry.dim
    vertices = [tuple(i[:dim]) for i in mesh.geometry.x]
    return vertices


def mesh_coordinates(
    mesh: Mesh,
    dim: int | None = None,
) -> tuple[np.ndarray, ...]:
    """
    `([x₀, x₁, x₂, ...], [y₀, y₁, y₂, ...], [z₀, z₁, z₂, ...])`
    """
    if dim is None:
        dim = mesh.geometry.dim
    return tuple(mesh.geometry.x[:, i] for i in range(dim))


@optional_lru_cache
def mesh_axes(
    mesh: Mesh,
    strict: bool = False,
)-> tuple[np.ndarray, ...]:
    """
    Unique and ordered `([x₀, x₁, x₂, ...], [y₀, y₁, y₂, ...], [z₀, z₁, z₂, ...])`
    """
    if strict and not is_cartesian(mesh):
        raise CartesianMeshError()
    return tuple(np.sort(np.unique(i)) for i in mesh_coordinates(mesh))


def mesh_vertices_tensor(
    mesh: Mesh,
    strict: bool = False,
) -> np.ndarray:
    """
    Tensor `v` such that `v[i, j]` returns the `(i, j)`th vertex of a structuted mesh. 
    """
    if strict and not is_cartesian(mesh):
        raise CartesianMeshError()
    mesh_axes = mesh_axes(mesh)
    if len(mesh_axes) == 2:
        x, y = mesh_axes
        return np.array([[(i, j) for j in y] for i in x])
    elif len(mesh_axes) == 3:
        x, y, z = mesh_axes
        return np.array([[[(i, j, k) for k in z] for j in y] for i in x])
    else:
        raise ValueError
    

def mesh_axes_spacing(
    mesh: Mesh,
    strict: bool = False,
) -> tuple[np.ndarray, ...]:
    """`([dx₀, dx₁, dx₂, ...], [dy₀, dy₁, dy₂, ...], [dz₀, dz₁, dz₂, ...])`"""
    if strict and not is_cartesian(mesh):
        raise CartesianMeshError()
    return tuple(np.diff(i) for i in mesh_axes(mesh))


@optional_lru_cache
def is_cartesian(
    mesh: Mesh,
) -> bool:
    axes = mesh_axes(mesh, strict=False)
    n_vertices = len(mesh.geometry.x)
    n_axes = [len(i) for i in axes]
    n_vertices_cartesian = np.prod(n_axes)
    return bool(n_vertices == n_vertices_cartesian)


@optional_lru_cache
def is_uniform_cartesian(
    mesh: Mesh,
) -> bool:
    if not is_cartesian(mesh):
        return False
    return all(all(np.isclose(dx[0], dx)) for dx in mesh_axes_spacing(mesh))


def number_of_entities(
    mesh: Mesh,
    dim: int,
    glob: bool = True,
    ghosts: bool = True,
) -> int:
    index_map = mesh.topology.index_map(dim)
    if glob:
        return index_map.size_global
    else:
        n = index_map.size_local
        if ghosts:
            n += index_map.num_ghosts
    return n


def number_of_cells(
    mesh: Mesh, 
    glob: bool = True,
    ghosts: bool = True,
) -> int:
    return number_of_entities(mesh, mesh.topology.dim, glob, ghosts)


def entities_to_coordinates(
    entity_indices: Iterable[int], 
    edim: int, 
    msh: Mesh, 
    centroid: bool = True,
) -> list[np.ndarray]:
    entity_indices = np.array(entity_indices, dtype=np.int32)
    vertex_indices = entities_to_geometry(msh, edim, entity_indices, False)

    coordinates = []
    for i in vertex_indices:
        vertex_coords = msh.geometry.x[i]
        if centroid:
            vertex_coords = np.mean(vertex_coords, axis=0)
        coordinates.append(vertex_coords)
    return coordinates


def cell_sizes(
    mesh: Mesh,
    h: str,
) -> np.ndarray:
    h = cell_size_quantity(mesh, h)
    return dofs(h, (mesh, 'DP', 0), use_cache=True)


def cell_aspect_ratios(mesh) -> np.ndarray:
    r = MaxCellEdgeLength(mesh) / MinCellEdgeLength(mesh)
    return dofs(r, (mesh, 'DP', 0), use_cache=True)


def cell_size_quantity(mesh: Mesh, h: str) -> GeometricCellQuantity:
    d = {
        'hmin': MinCellEdgeLength,
        'hmax': MaxCellEdgeLength,
        'hdiam': CellDiameter,
        'hrad': Circumradius,
    }
    try:
        return d[h](mesh)
    except KeyError:
        raise ValueError(f'Invalid cell size quantity {h}.')
    

def CartesianMeshError():
    return ValueError("Mesh vertices must form a Cartesian grid.")
