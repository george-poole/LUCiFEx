from typing import Literal, ParamSpec, TypeVar, Callable, overload
from collections.abc import Iterable
from functools import wraps

import numpy as np
from ufl import Measure
from ufl.core.expr import Expr
from dolfinx.fem import assemble_scalar, form
from dolfinx.mesh import Mesh, locate_entities, meshtags

from .fem_utils import extract_mesh
from .dofs_utils import (
    as_spatial_marker,
    Marker,
)


def create_tagged_measure(
    measure: Literal['dx', 'ds', 'dS'],
    mesh: Mesh,
    markers: Iterable[Marker] = (),
    tags: Iterable[int] | None = None,
    tag_unmarked: int | None = None,
    **metadata,
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
        return Measure(measure, domain=mesh, metadata=metadata)
    
    if tags is None:
        marker_tags = list(range(len(markers)))
    if tag_unmarked is None:
        tag_unmarked = max(marker_tags) + 1
    assert tag_unmarked not in marker_tags

    gdim = mesh.topology.dim
    fdim = gdim - 1
    mesh.topology.create_entities(fdim)
    facet_index_map = mesh.topology.index_map(fdim)
    num_facets = facet_index_map.size_local + facet_index_map.num_ghosts
    facet_tags = np.full(num_facets, tag_unmarked, dtype=np.intc)

    for t, m in zip(marker_tags, markers, strict=True):
        m = as_spatial_marker(m)
        facet_indices = locate_entities(mesh, fdim, m)
        facet_tags[facet_indices] = t

    mesh_tags = meshtags(mesh, fdim, np.arange(num_facets), facet_tags)
    return Measure(measure, domain=mesh, subdomain_data=mesh_tags, metadata=metadata)    

    
P = ParamSpec('P')
R = TypeVar('R') # Expr | tuple[Expr, ...]
def integral(
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
        *markers: Marker,
        facet_side: Literal['+', '-'] | None = None,
        **metadata,
    ) -> Callable[P, float | np.ndarray]:
        ...


    def _overload(
        measure: Literal['dx', 'ds', 'dS'] | Measure, 
        *markers: Marker,
        facet_side: Literal['+', '-'] | None = None,
        **metadata,
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
                dx_tagged = create_tagged_measure(measure, mesh, markers, **metadata)

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


