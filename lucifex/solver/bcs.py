from typing import TypeAlias
from collections.abc import Iterable
from typing_extensions import Unpack

from ufl import Measure, Form, TestFunction, inner
from ufl.core.expr import Expr
from dolfinx_mpc import MultiPointConstraint
from dolfinx.fem import (
    Function,
    FunctionSpace,
    Constant,
    DirichletBCMetaClass,
    dirichletbc,
)

from ..utils.fem_utils import is_scalar, is_vector
from ..utils.enum_types import BoundaryType
from ..utils.fem_typecasting import fem_function, fem_constant
from ..utils.measure_utils import create_tagged_measure
from ..utils.dofs_utils import (
    as_spatial_marker,
    dofs_indices,
    Marker,
    SpatialExpression,
    SubspaceIndex,
    DofsMethodType,
)

Value: TypeAlias = (
    Function
    | Constant
    | SpatialExpression
    | float
    | Iterable[float]
    | Expr
)


class BoundaryConditions:
    """
    The `'neumann'` boundary type with boundary value `g` assumes a form
    `+∫ v·g ds` for the boundary term in the variational formulation.

    More complicated boundary terms should instead be specified in the forms function.
    """
    def __init__(
        self,
        *bcs: tuple[BoundaryType, Marker, Value]
        | tuple[BoundaryType, Marker, Value, SubspaceIndex]
        | tuple[Marker, Value]
        | tuple[Marker, Value, SubspaceIndex],
        dofs_method: DofsMethodType | Iterable[DofsMethodType] = DofsMethodType.TOPOLOGICAL,
    ):
        self._markers: list[Marker] = []
        self._values: list = []
        self._btypes: list[BoundaryType] = []
        self._subindices: list[int | None] = []
        
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
            self._markers.append(marker)
            self._values.append(value)
            self._btypes.append(BoundaryType(btype))
            self._subindices.append(subindex)
            
    def create_strong_bcs(
        self,
        function_space: FunctionSpace,
    ) -> list[DirichletBCMetaClass]: 
        """
        Strongly enforced boundary condition `u = uD` on `∂Ω`
        """
        dirichlet = []

        for uD, b, m, i, d in zip(
            self._values, self._btypes, self._markers, self._subindices, self._dofs_method,
            strict=True,
        ):
            if b in (BoundaryType.DIRICHLET, BoundaryType.ESSENTIAL):
                dofs = dofs_indices(function_space, m, i, d)
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

    def create_periodic_bcs(
        self,
        function_space: FunctionSpace,
        bcs: list[DirichletBCMetaClass] | None = None,
        finalize: bool = True,
    ) -> MultiPointConstraint | None:
        """
        Implements periodic and antiperiodic boundary conditions via a geometrical constraint
        """
    
        if bcs is None:
            bcs = []
        
        mpc = MultiPointConstraint(function_space)
        n_constraint = 0

        for reln, b, m, i in zip(self._values, self._btypes, self._markers, self._subindices):
            if b in (BoundaryType.PERIODIC, BoundaryType.ANTIPERIODIC):
                if i is not None:
                    raise NotImplementedError
                scale = 1.0 if b == BoundaryType.PERIODIC else -1.0
                mpc.create_periodic_constraint_geometrical(
                    function_space, 
                    as_spatial_marker(m),
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
        function_space: FunctionSpace,
    ) -> list[Form]:
        """
        Weakly enforced boundary condition by a term `+∫ v·uN ds` with 
        test function `v` and prescribed value `uN` in the variational form `F(v,u)=0`.
        """
        
        v = TestFunction(function_space)
        boundary_types = (
            BoundaryType.NEUMANN, BoundaryType.ROBIN, 
            BoundaryType.NATURAL, BoundaryType.WEAK_DIRICHLET,
        )
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

        tag = 0
        tags = {b: [] for b in boundary_types}
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

                tags[b].append(tag)
                markers[b].append(m)
                exprs[b].append(g)
                tag += 1

        nums_all = [t for _tags in tags.values() for t in _tags]
        markers_all = [m for _markers in markers.values() for m in _markers]
        ds = create_tagged_measure('ds', function_space.mesh, markers_all, nums_all)
        tags_exprs = [[(t, e) for t, e in zip(tags[b], exprs[b])] for b in boundary_types]

        return ds, *tags_exprs

