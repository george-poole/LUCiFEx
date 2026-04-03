from typing import TypeAlias, Callable, TypeAlias
from collections.abc import Iterable
from typing_extensions import Unpack

import numpy as np
from ufl import Measure, Form, Argument, TestFunction, inner, replace, TrialFunction
from ufl.core.expr import Expr
from dolfinx_mpc import MultiPointConstraint
from dolfinx.fem import (
    FunctionSpace,
    DirichletBCMetaClass,
    dirichletbc,
)

from ..fdm import FunctionSeries
from ..fem import Function, Constant
from ..utils.py_utils import ToDoError, AnyFloat
from ..utils.fenicsx_utils import (
    BoundaryType, 
    create_tagged_measure,
    as_boolean_marker,
    dofs_indices,
    Marker,
    BooleanMarker,
    DofsLocatorType,
    FacetLocatorType,
    as_function, 
    create_function, create_constant, 
    extract_function_space,
)

Value: TypeAlias = (
    Function
    | Expr
    | Constant
    | Callable[[np.ndarray], np.ndarray]
    | AnyFloat
    | Iterable[AnyFloat]
)

SubspaceIndex: TypeAlias = int | None 


V: TypeAlias = Argument
UW: TypeAlias = Expr | Function | Constant
class BoundaryConditions:
    """
    A weakly-enforced boundary condition (e.g. `'natural'`, `'neumann'` or `'robin'`) with 
    boundary value `uW` will add to the variational forms a term `+∫ f(v,uW) ds` , given 
    the test function `v` and some expression factory `f` which if unspecified 
    defaults to `f(v,uW) = v·uW`.
    """
    def __init__(
        self,
        *bcs: tuple[BoundaryType, Marker, Value]
        | tuple[BoundaryType, Marker, Value, SubspaceIndex]
        | tuple[Marker, Value]
        | tuple[Marker, Value, SubspaceIndex],
        dofs_locator: DofsLocatorType | Iterable[DofsLocatorType] = DofsLocatorType.TOPOLOGICAL,
        **auto_exprs: Callable[[V, UW], Expr], 
    ):
        self._markers: list[BooleanMarker] = []
        self._values: list = []
        self._btypes: list[BoundaryType] = []
        self._subindices: list[int | None] = []

        if isinstance(dofs_locator, str):
            dofs_locator = [dofs_locator] * len(bcs)
        assert len(dofs_locator) == len(bcs)
        self._dofs_locator = [DofsLocatorType(i) for i in dofs_locator]

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
            self._markers.append(as_boolean_marker(marker))
            self._values.append(value)
            self._btypes.append(BoundaryType(btype))
            self._subindices.append(subindex)

        _weak_btypes = (BoundaryType.NEUMANN, BoundaryType.WEAK, BoundaryType.NATURAL, BoundaryType.ROBIN)
        _auto_exprs: dict[BoundaryType, Callable[[V, UW], Expr]] = {
            btype: lambda v, uW: inner(v, uW)
            for btype in _weak_btypes
        }
        _auto_exprs.update({BoundaryType(k): v for k, v in auto_exprs.items()})
        self._auto_exprs = _auto_exprs
            
    def create_strong_bcs(
        self,
        fs: FunctionSpace,
        strong_types: Iterable[BoundaryType] = (
            BoundaryType.DIRICHLET, BoundaryType.ESSENTIAL, BoundaryType.STRONG,
        ),
        facet_locator: FacetLocatorType = FacetLocatorType.BOUNDARY,
        fs_sub: Iterable[FunctionSpace | None] | None = None,
    ) -> list[DirichletBCMetaClass]: 
        """
        Strongly enforced boundary condition `u = uE` on `∂Ω`
        """
        dirichlet = []

        for uD, b, m, i, d in zip(
            self._values, self._btypes, self._markers, self._subindices, self._dofs_locator,
            strict=True,
        ):
            if b in strong_types:
                if fs_sub is None:
                    dofs = dofs_indices(fs, m, i, d, facet_locator)
                else:
                    dofs = dofs_indices(fs_sub[i], m, dofs_locator=d, facet_locator=facet_locator, collapsed=True)
                if isinstance(uD, Constant):
                    if fs_sub is None:
                        dbc = dirichletbc(uD, dofs, fs if i is None else fs.sub(i))
                    else:
                        dbc = dirichletbc(uD, dofs, fs_sub[i]) # TODO check this
                else:
                    if fs_sub is None:
                        uD = as_function(fs, uD, i)
                        dbc = dirichletbc(uD, dofs, None if i is None else fs.sub(i))
                    else:
                        uD = as_function(fs_sub[i], uD)
                        dbc = dirichletbc(uD, dofs)
                dirichlet.append(dbc)
                
        return dirichlet

    def create_periodic_mpc(
        self,
        fs: FunctionSpace,
        bcs: list[DirichletBCMetaClass] | None = None,
        finalize: bool = True,
    ) -> MultiPointConstraint | None:
        """
        Implements periodic and antiperiodic boundary conditions via a geometrical constraint.
        """
    
        if bcs is None:
            bcs = []
        
        mpc = MultiPointConstraint(fs)
        n_constraint = 0

        for reln, b, m, i in zip(self._values, self._btypes, self._markers, self._subindices):
            if b in (BoundaryType.PERIODIC, BoundaryType.ANTIPERIODIC):
                if i is not None:
                    raise ToDoError
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
        fs_or_u: Function | FunctionSeries | FunctionSpace,
        weak_types: Iterable[BoundaryType] = (
            BoundaryType.NEUMANN, BoundaryType.ROBIN, 
            BoundaryType.NATURAL, BoundaryType.WEAK,
        )
    ) -> list[Form]:
        fs = extract_function_space(fs_or_u)
        v = TestFunction(fs)

        ds, *u_weaks = self.boundary_data(fs_or_u, *weak_types)
        forms = []
        
        for u_weak, tp in zip(u_weaks, weak_types, strict=True):
            if u_weak:
                expr = self._auto_exprs[tp]
                F_weak = sum([expr(v, uW) * ds(i) for i, uW in u_weak])
                forms.append(F_weak)

        return forms
    
    def boundary_data(
        self,
        fs_or_u: FunctionSpace | Function | FunctionSeries | Argument,
        *boundary_types: BoundaryType,
        strict: bool = False,
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
        fs = extract_function_space(fs_or_u) 
        boundary_types = [BoundaryType(i) for i in boundary_types]

        if strict and not all(b in boundary_types for b in self._btypes):
            raise ValueError

        tag = 0
        tags = {b: [] for b in boundary_types}
        markers = {b: [] for b in boundary_types}
        values = {b: [] for b in boundary_types}

        for b, m, uB, i in zip(
            self._btypes, self._markers, self._values, self._subindices, 
            strict=True,
        ):
            if b == BoundaryType.ROBIN and isinstance(fs_or_u, (Function, FunctionSeries)):
                uB = self.robin_auto_expr(uB, fs_or_u, future=True)
            if b in boundary_types:
                if isinstance(uB, (Function, Expr, Constant)):
                    _uB = uB
                elif isinstance(uB, Iterable):
                    if all(isinstance(gi, (float, int)) for gi in uB):
                        _uB = create_constant(fs.mesh, uB)
                    else:
                        _uB = create_function(fs, uB, i)
                elif isinstance(uB, (float, int)):
                    _uB = create_constant(fs.mesh, uB) 
                else:
                    _uB = create_function(fs, uB, i)
                tags[b].append(tag)
                markers[b].append(m)
                values[b].append(_uB)
                tag += 1

        tags_all = [t for _tags in tags.values() for t in _tags]
        markers_all = [m for _markers in markers.values() for m in _markers]
        ds = create_tagged_measure('ds', fs.mesh, markers_all, tags_all)
        tags_values = [[(t, e) for t, e in zip(tags[b], values[b])] for b in boundary_types]

        return ds, *tags_values
    
    @staticmethod
    def robin_auto_expr(
        expr: Expr,
        u: Function | FunctionSeries,
        future: bool | int,
    ) -> Expr:
        fs = u.function_space
        if isinstance(u, Function):
            return replace(expr, {u: TrialFunction(fs)})
        else:
            if future is True:
                index = u.FUTURE_INDEX
            elif future is False:
                index = u.FUTURE_INDEX - 1
            else:
                index = future
            return replace(expr, {u[index]: TrialFunction(fs)})
    
