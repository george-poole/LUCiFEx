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
    The `'neumann'` boundary type with boundary value `g` assumes a form
    `+∫ v·g ds` for the boundary term in the variational formulation

    More complicated boundary terms should instead be specified in the forms function.
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

    # TODO test all cases
    def create_strong_bcs(
        self,
        function_space: FunctionSpace,
    ) -> list[DirichletBCMetaClass]: 
        """
        Strongly enforced boundary condition `u = uD` on `∂Ω`
        """
        
        dirichlet = []

        for uD, b, m, i in zip(
            self._values, self._btypes, self._markers, self._subindices, 
            strict=True,
        ):
            if b in (BoundaryType.DIRICHLET, BoundaryType.ESSENTIAL):
                dofs = dofs_indices(function_space, m, i)
                if isinstance(uD, Constant):
                    if i is None:
                        dbc = dirichletbc(uD, dofs, function_space)
                    else:
                        dbc = dirichletbc(uD, dofs, function_space.sub(i))
                else:
                    uD = fem_function(function_space, uD, i, try_identity=True)
                    if i is None:
                        dbc = dirichletbc(uD, dofs)
                    else:
                        dbc = dirichletbc(uD, dofs, function_space.sub(i))
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
        """
        Weakly enforced boundary condition by a term `+∫ v·uN ds` with 
        test function `v` and prescribed value `uN` in the variational form `F(v,u)=0`.
        """
        
        v = TestFunction(function_space)
        boundary_types = (BoundaryType.NEUMANN, BoundaryType.ROBIN, BoundaryType.NATURAL)
        ds, *boundary_data = self.boundary_data(function_space, *boundary_types)

        forms = []
        
        for bd in boundary_data:
            for i, uN in bd:
                if is_scalar(uN):
                    forms.append(v * uN * ds(i))
                elif is_vector(uN):
                    forms.append(inner(v, uN) * ds(i))
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
        `ds, [(0, f₀), (1, f₁), (2, f₂), ...]` if one boundary type is given \\
        `ds, [(0, f₀), ...], [(1, f₁), ...]` if two boundary types given \\
        `ds, [(0, f₀), ...], [(1, f₁), ...], [(2, f₂), ...]` if three boundary types given \\
        etc.

        `ds(n)` is the measure for subset for the boundary where the `n`th boundary condition.
        
        Given `n_total` boundary conditions, `ds(n_total)` is the measure for the subset of the boundary 
        where no boundary conditions are applied.
        """
        
        boundary_types = [BoundaryType(i) for i in boundary_types]

        num = 0
        nums = {b: [] for b in boundary_types}
        exprs = {b: [] for b in boundary_types}
        markers = {b: [] for b in boundary_types}

        for b, m, g, i in zip(
            self._btypes, self._markers, self._values, self._subindices, 
            strict=True,
        ):
            if b in boundary_types:
                if isinstance(g, (Function, Expr, Constant)):
                    pass 
                elif isinstance(g, Iterable):
                    if all(isinstance(gi, (float, int)) for gi in g):
                        g = fem_constant(function_space.mesh, g)
                    else:
                        g = fem_function(function_space, g, i)
                elif isinstance(g, (float, int)):
                    g = fem_constant(function_space.mesh, g)
                else:
                    g = fem_function(function_space, g, i)

                nums[b].append(num)
                exprs[b].append(g)
                markers[b].append(m)
                num += 1

        nums_all = [n for ns in nums.values() for n in ns]
        markers_all = [m for ms in markers.values() for m in ms]
        ds = create_enumerated_measure('ds', function_space.mesh, markers_all, nums_all)
        nums_markers_all = [[(t, e) for t, e in zip(nums[b], exprs[b])] for b in boundary_types]

        return ds, *nums_markers_all


def create_enumerated_measure(
    measure: Literal['dx', 'ds', 'dS'],
    mesh: Mesh,
    markers: Iterable[SpatialMarker] = (),
    nums: Iterable[int] | None = None,
    num_unmarked: int | None = None,
) -> Measure:
    if len(markers) == 0:
        return Measure(measure, domain=mesh)
    
    if nums is None:
        nums = list(range(len(markers)))

    if num_unmarked is None:
        num_unmarked = max(nums) + 1

    assert num_unmarked not in nums

    gdim = mesh.topology.dim
    fdim = gdim - 1
    mesh.topology.create_entities(fdim)
    facet_index_map = mesh.topology.index_map(fdim)
    num_facets = facet_index_map.size_local + facet_index_map.num_ghosts
    facet_indices_sorted = np.arange(num_facets)
    facet_tags = np.arange(num_facets, dtype=np.intc)
    facet_indices_tagged = []

    for t, m in zip(nums, markers, strict=True):
        m = as_spatial_indicator_func(m)
        facet_indices = locate_entities(mesh, fdim, m)
        facet_tags[facet_indices] = t
        facet_indices_tagged.extend(facet_indices)

    facet_indices_unmarked = set(facet_indices_sorted).difference(
        facet_indices_tagged
    )
    facet_tags[list(facet_indices_unmarked)] = num_unmarked

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
