from typing import TypeAlias, Literal
from collections.abc import Iterable
from typing_extensions import Unpack

import numpy as np
from ufl import Measure, Form, TestFunction, inner
from ufl.core.expr import Expr
from dolfinx_mpc import MultiPointConstraint
from dolfinx.mesh import Mesh, locate_entities, meshtags
from dolfinx.fem import (
    Function,
    FunctionSpace,
    Constant,
    Expression,
    DirichletBCMetaClass,
    dirichletbc,
)

from ..utils.fem_utils import is_scalar, is_vector
from ..utils.enum_types import BoundaryType
from ..utils.fem_typecasting import fem_function, fem_constant
from ..utils.dofs_utils import (
    as_spatial_indicator_func,
    dofs_indices,
    SpatialMarker,
    SpatialExpressionFunc,
    SubspaceIndex,
)

Value: TypeAlias = (
    Function
    | Constant
    | SpatialExpressionFunc
    | float
    | Iterable[float]
    | Expr
)

# TODO understand master-slaves dof relations, single vs double periodic bc
def periodic_relation(m: SpatialMarker, v: SpatialMarker): 
    def _rltn(x: np.ndarray):
        x_master = x.copy()
        x_master = m(v(x))
        return x_master
    return _rltn


class BoundaryConditions:
    """
    For a boundary term `∫ v·g ds` in the variational formulation specify
    a `'neumann'` boundary type and supply the expression `g`.
    """
    def __init__(
        self,
        *bcs: tuple[BoundaryType, SpatialMarker, Value]
        | tuple[BoundaryType, SpatialMarker, Value, SubspaceIndex]
        | tuple[SpatialMarker, Value]
        | tuple[SpatialMarker, Value, SubspaceIndex],
    ):
        self._markers: list[SpatialMarker] = []
        self._values: list = []
        self._btypes: list[BoundaryType] = []
        self._subindices: list[int | None] = []

        for bc in bcs:
            match bc:
                case marker, value:
                    marker, value = bc
                    btype = BoundaryType.DIRICHLET
                    subindex = None
                case marker, value, subindex if isinstance(subindex, int):
                    marker, value, subindex = bc
                    btype = BoundaryType.DIRICHLET
                case btype, marker, value: 
                    btype, marker, value = bc
                    subindex = None
                case btype, marker, value, subindex:
                    btype, marker, value, subindex = bc
                case _:
                    raise ValueError(f"{bc} not a valid boundary condition")
            self._markers.append(marker)
            self._values.append(value)
            self._btypes.append(BoundaryType(btype))
            self._subindices.append(subindex)

    def create_strong_bcs(
        self,
        function_space: FunctionSpace,
    ) -> list[DirichletBCMetaClass]: 
        
        dirichlet = []

        for b, g, m, i in zip(
            self._btypes, self._values, self._markers, self._subindices, 
            strict=True,
        ):
            if b in (BoundaryType.DIRICHLET, BoundaryType.ESSENTIAL):
                g = fem_function(function_space, g, i)
                dofs = dofs_indices(function_space, m, i)
                if i is not None:
                    dbc = dirichletbc(g, dofs, function_space.sub(i))
                    # TODO check this is incorrect dbc = dirichletbc(g, dofs[0])
                else:
                    dbc = dirichletbc(g, dofs)
                dirichlet.append(dbc)
                
        return dirichlet


    # TODO fix and test
    def create_periodic_bcs(
        self,
        function_space: FunctionSpace,
        bcs: list[DirichletBCMetaClass] | None = None,
    ) -> MultiPointConstraint | None:
    
        if bcs is None:
            bcs = self.create_strong_bcs(function_space)
        
        mpc = MultiPointConstraint(function_space)
        n_constraint = 0

        for b, m, g in zip(self._btypes, self._markers, self._values):
            if b in (BoundaryType.PERIODIC, BoundaryType.ANTIPERIODIC):
                g = as_spatial_indicator_func(g)
                mpc.create_periodic_constraint_geometrical(
                    function_space, m, periodic_relation(m, g), bcs
                )
                n_constraint += 1

        if n_constraint == 0:
            return None
        else:
            mpc.finalize()
            return mpc


    def create_weak_bcs(
        self,
        function_space: FunctionSpace,
    ) -> list[Form]:
        """ Assumes a weak term of the form `∫ v·g ds` with test function `v` and
        prescribed value `g`.
        
        More complicated forms should be specified in the forms function.
        """
        
        v = TestFunction(function_space)
        boundary_types = (BoundaryType.NEUMANN, BoundaryType.ROBIN, BoundaryType.NATURAL)
        ds, *boundary_data = self.boundary_data(function_space, *boundary_types)

        forms = []
        
        for bd in boundary_data:
            for i, g in bd:
                if is_scalar(g):
                    forms.append(v * g * ds(i))
                elif is_vector(g):
                    forms.append(inner(v, g) * ds(i))
                else:
                    raise NotImplementedError

        return forms
    

    def boundary_data(
        self,
        function_space: FunctionSpace,
        *boundary_types: BoundaryType,
    ) -> tuple[Measure, Unpack[tuple[list[tuple[int, Constant | Function | Expr]], ...]]]:
        """
        Returns \\
        `ds, [(0, f₀), (1, f₁), (2, f₂), ...]` if one boundary types is given \\
        `ds, [(0, f₀), ...], [(1, f₁), ...]` if two boundary types given \\
        `ds, [(0, f₀), ...], [(1, f₁), ...], [(2, f₂), ...]` if three boundary types given \\
        etc.
        """
        
        boundary_types = [BoundaryType(i) for i in boundary_types]

        tag = 0
        tags = {b: [] for b in boundary_types}
        exprs = {b: [] for b in boundary_types}
        markers = {b: [] for b in boundary_types}

        for b, m, g, i in zip(
            self._btypes, self._markers, self._values, self._subindices, 
            strict=True,
        ):
            if b in boundary_types:
                if isinstance(g, Expr):
                    pass 
                elif isinstance(g, Iterable):
                    if all(isinstance(gi, (float, int)) for gi in g):
                        g = fem_constant(function_space.mesh, g)
                    else:
                        g = fem_function(function_space, g, i)
                elif isinstance(g, (float, int, Constant)):
                    g = fem_constant(function_space.mesh, g)
                else:
                    g = fem_function(function_space, g, i)

                tags[b].append(tag)
                exprs[b].append(g)
                markers[b].append(m)
                tag += 1

        tags_all = [t for ts in tags.values() for t in ts]
        markers_all = [m for ms in markers.values() for m in ms]
        ds = create_marked_measure('ds', function_space.mesh, markers_all, tags_all)
        tags_markers_all = [[(t, e) for t, e in zip(tags[b], exprs[b])] for b in boundary_types]

        return ds, *tags_markers_all


def create_marked_measure(
    measure: Literal['dx', 'ds', 'dS'],
    mesh: Mesh,
    markers: Iterable[SpatialMarker] = (),
    tags: Iterable[int] | None = None,
    tag_unmarked: int | None = None,
) -> Measure:
    if len(markers) == 0:
        return Measure(measure, domain=mesh)
    
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
    facet_indices_sorted = np.arange(num_facets)
    facet_tags = np.arange(num_facets, dtype=np.intc)
    facet_indices_marked = []

    for t, m in zip(tags, markers, strict=True):
        m = as_spatial_indicator_func(m)
        facet_indices = locate_entities(mesh, fdim, m)
        facet_tags[facet_indices] = t
        facet_indices_marked.extend(facet_indices)

    facet_indices_unmarked = set(facet_indices_sorted).difference(
        facet_indices_marked
    )
    facet_tags[list(facet_indices_unmarked)] = tag_unmarked

    mesh_tags = meshtags(mesh, fdim, facet_indices_sorted, facet_tags)
    return Measure(measure, domain=mesh, subdomain_data=mesh_tags)


# def create_marked_dx(
#     mesh: Mesh,
#     markers: list[SpatialMarker] | None = None,
#     tags: list[int] | None = None,
# ) -> ufl.Measure:
#     if not markers is None:
#         return ufl.Measure("dx", domain=mesh)
    
#     if tags is None:
#         tags = list(range(len(markers)))

#     for t, m in zip(tags, markers, strict=True):
#         m = as_spatial_indicator_func(m)
#         cell_indices = locate_entities(mesh, mesh.topology.dim, m)
#         raise NotImplementedError  # TODO

#     mesh_tags = ...
#     return ufl.Measure("dx", domain=mesh, subdomain_data=mesh_tags)


# # alternative method 1 - this gets facets for the ENTIRE mesh boundary
# function_space.mesh.domain.topology.create_connectivity(edim, tdim)
# facets = dfx.mesh.exterior_facet_indices(function_space.mesh.topology)

# # alernative method 2 - is this pointless when considering one (vs many) BC??
# facet_indices = dfx.mesh.locate_entities(
#     function_space.mesh, edim, self._marker
# )
# facet_indices = np.argsort(facet_indices)
# facet_ids = np.full_like(facets, boundary_id)
# facet_tags = dfx.mesh.meshtags(
#     function_space.mesh, edim, facet_indices, facet_ids
# )
# facets = facet_tags.find(boundary_id)
