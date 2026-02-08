from typing import TypeAlias, Callable
from collections.abc import Iterable
from typing_extensions import Unpack

import numpy as np
from ufl import Measure, Form, TestFunction, inner, replace, TrialFunction
from ufl.core.expr import Expr
from dolfinx_mpc import MultiPointConstraint
from dolfinx.fem import (
    FunctionSpace,
    DirichletBCMetaClass,
    dirichletbc,
)

from ..fdm import FunctionSeries
from ..fem import Function, Constant
from ..utils.fem_utils import is_scalar, is_vector, create_fem_function, create_fem_constant
from ..utils.mesh_utils import BoundaryType, create_tagged_measure
from ..utils.dofs_utils import (
    as_spatial_marker,
    dofs_indices,
    SpatialMarkerAlias,
    SpatialMarker,
    SubspaceIndex,
    DofsMethodType,
    FacetMethodType,
    ScalarVectorError,
)

Value: TypeAlias = (
    Function
    | Constant
    | Callable[[np.ndarray], np.ndarray]
    | float
    | Iterable[float]
    | Expr
)


class BoundaryConditions:
    """
    A weakly-enforced boundary condition (e.g. `'natural'`, `'neumann'` or `'robin'`) with 
    boundary value `uW` will add to the variational forms a term `+∫ v·uW ds`, 
    where `v` is the test function.

    Weakly-enforced boundary conditions that cannnot be expressed by 
    such a form should instead be specified in the forms function.
    """
    def __init__(
        self,
        *bcs: tuple[BoundaryType, SpatialMarker | SpatialMarkerAlias, Value]
        | tuple[BoundaryType, SpatialMarker | SpatialMarkerAlias, Value, SubspaceIndex]
        | tuple[SpatialMarker | SpatialMarkerAlias, Value]
        | tuple[SpatialMarker | SpatialMarkerAlias, Value, SubspaceIndex],
        dofs_method: DofsMethodType | Iterable[DofsMethodType] = DofsMethodType.TOPOLOGICAL,
        rescale_weak: int | float | None = None, 
    ):
        self._markers: list[SpatialMarker] = []
        self._values: list = []
        self._btypes: list[BoundaryType] = []
        self._subindices: list[int | None] = []
        self._rescale_weak = rescale_weak

        if isinstance(dofs_method, str):
            dofs_method = [dofs_method] * len(bcs)
        assert len(dofs_method) == len(bcs)
        self._dofs_method = [DofsMethodType(i) for i in dofs_method]

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
            self._markers.append(as_spatial_marker(marker))
            self._values.append(value)
            self._btypes.append(BoundaryType(btype))
            self._subindices.append(subindex)
            
    def create_strong_bcs(
        self,
        fs: FunctionSpace,
        strong_types: Iterable[BoundaryType] = (
            BoundaryType.DIRICHLET, BoundaryType.ESSENTIAL, BoundaryType.STRONG,
        ),
        facet_method: FacetMethodType = FacetMethodType.BOUNDARY,
    ) -> list[DirichletBCMetaClass]: 
        """
        Strongly enforced boundary condition `u = uE` on `∂Ω`
        """
        dirichlet = []

        for uD, b, m, i, d in zip(
            self._values, self._btypes, self._markers, self._subindices, self._dofs_method,
            strict=True,
        ):
            if b in strong_types:
                dofs = dofs_indices(fs, m, i, d, facet_method=facet_method)
                if isinstance(uD, Constant):
                    if i is None:
                        dbc = dirichletbc(uD, dofs, fs)
                    else:
                        dbc = dirichletbc(uD, dofs, fs.sub(i))
                else:
                    uD = create_fem_function(fs, uD, i, try_identity=True)
                    if i is None:
                        dbc = dirichletbc(uD, dofs)
                    else:
                        dbc = dirichletbc(uD, dofs, fs.sub(i))
                dirichlet.append(dbc)
                
        return dirichlet

    def create_periodic_bcs(
        self,
        fs: FunctionSpace,
        bcs: list[DirichletBCMetaClass] | None = None,
        finalize: bool = True,
    ) -> MultiPointConstraint | None:
        """
        Implements periodic and antiperiodic boundary conditions via a geometrical constraint
        """
    
        if bcs is None:
            bcs = []
        
        mpc = MultiPointConstraint(fs)
        n_constraint = 0

        for reln, b, m, i in zip(self._values, self._btypes, self._markers, self._subindices):
            if b in (BoundaryType.PERIODIC, BoundaryType.ANTIPERIODIC):
                if i is not None:
                    raise NotImplementedError
                scale = 1.0 if b == BoundaryType.PERIODIC else -1.0
                mpc.create_periodic_constraint_geometrical(
                    fs, 
                    m,
                    reln,
                    bcs=bcs,
                    scale=scale,
                )
                n_constraint += 1

        if n_constraint == 0:
            return None
        else:
            if finalize:
                mpc.finalize()
            return mpc

    def create_weak_bcs(
        self,
        solution: Function | FunctionSeries | FunctionSpace,
        weak_types: Iterable[BoundaryType] = (
            BoundaryType.NEUMANN, BoundaryType.ROBIN, 
            BoundaryType.NATURAL, BoundaryType.WEAK,
        )
    ) -> list[Form]:
        """
        Weakly enforced boundary condition by a term `+∫ v·uW ds` with 
        test function `v` and prescribed value `uW` in the variational form `F(u,v)=0`.
        """
        fs = solution.function_space
        v = TestFunction(fs)
        if self._rescale_weak is not None:
            scale = Constant(fs.mesh, self._rescale_weak)
        else:
            scale = 1

        ds, *boundary_data = self.boundary_data(solution, *weak_types)
        forms = []
        
        for bd in boundary_data:
            for i, uW in bd:
                if is_scalar(uW):
                    forms.append(scale * v * uW * ds(i))
                elif is_vector(uW):
                    forms.append(scale * inner(v, uW) * ds(i))
                else:
                    raise ScalarVectorError(uW)

        return forms
    
    def boundary_data(
        self,
        solution: Function | FunctionSeries | FunctionSpace,
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
        if not isinstance(solution, FunctionSpace):
            fs = solution.function_space
        else:
            fs = solution
            
        boundary_types = [BoundaryType(i) for i in boundary_types]
        tag = 0
        tags = {b: [] for b in boundary_types}
        exprs = {b: [] for b in boundary_types}
        markers = {b: [] for b in boundary_types}

        for b, m, g, i in zip(
            self._btypes, self._markers, self._values, self._subindices, 
            strict=True,
        ):
            if b == BoundaryType.ROBIN and not isinstance(solution, FunctionSpace):
                g = create_robin(g, solution)
            if b in boundary_types:
                if isinstance(g, (Function, Expr, Constant)):
                    pass 
                elif isinstance(g, Iterable):
                    if all(isinstance(gi, (float, int)) for gi in g):
                        g = create_fem_constant(fs.mesh, g)
                    else:
                        g = create_fem_function(fs, g, i)
                elif isinstance(g, (float, int)):
                    g = create_fem_constant(fs.mesh, g)
                else:
                    g = create_fem_function(fs, g, i)

                tags[b].append(tag)
                markers[b].append(m)
                exprs[b].append(g)
                tag += 1

        nums_all = [t for _tags in tags.values() for t in _tags]
        markers_all = [m for _markers in markers.values() for m in _markers]
        ds = create_tagged_measure('ds', fs.mesh, markers_all, nums_all)
        tags_exprs = [[(t, e) for t, e in zip(tags[b], exprs[b])] for b in boundary_types]

        return ds, *tags_exprs


def create_robin(
    expr: Expr,
    solution: Function | FunctionSeries,
    future: bool = True,
) -> Expr:
    fs = solution.function_space
    if isinstance(solution, Function):
        return replace(expr, {solution: TrialFunction(fs)})
    else:
        if future:
            index = solution.FUTURE_INDEX
        else:
            index = solution.FUTURE_INDEX - 1
        return replace(expr, {solution[index]: TrialFunction(fs)})
    
