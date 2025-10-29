from typing import overload

import numpy as np
from dolfinx.mesh import Mesh, locate_entities_boundary

from ..utils.dofs_utils import SpatialMarker, SpatialMarkerAlias, as_spatial_marker


class MeshBoundary:
    def __init__(self, boundaries: dict[str | int, SpatialMarker | SpatialMarkerAlias]):
        self._boundaries = boundaries

    @overload
    def __getitem__(
        self, 
        tag: str | int,
    ) -> SpatialMarker | SpatialMarkerAlias:
        ...

    @overload
    def __getitem__(
        self, 
        tag: tuple[str | int, ...],
    ) -> list[SpatialMarker | SpatialMarkerAlias]:
        ...

    def __getitem__(
        self, 
        tag: str | int | tuple[str | int, ...],
    ) -> SpatialMarker | SpatialMarkerAlias:
        if isinstance(tag, tuple):
            return [self._boundaries[t] for t in tag]
        else:
            return self._boundaries[tag]

    @property
    def union(self) -> list[SpatialMarker | SpatialMarkerAlias]:
        """
        `∪ᵢ∂Ωᵢ`
        """
        return list(self._boundaries.values())
    
    @property
    def union_difference(self) -> SpatialMarker:
        """
        `∂Ω \ ∪ᵢ∂Ωᵢ`

        If the defined boundaries are complete, then `∂Ω = ∪ᵢ∂Ωᵢ` and hence `∂Ω \ ∪ᵢ∂Ωᵢ = ∅`.
        """
        on_boundaries = as_spatial_marker(self.union)
        return lambda x: np.logical_not(on_boundaries(x))
    
    @property
    def names(self) -> tuple[str | int, ...]:
        return tuple(self._boundaries.keys())
    
    @property
    def markers(self) -> tuple[SpatialMarker | SpatialMarkerAlias, ...]:
        return tuple(self._boundaries.values())


def mesh_boundary(
    mesh: Mesh,
    boundaries: dict[str | int, SpatialMarker | SpatialMarkerAlias],
    verify: bool = True,
    complete: bool = False,
) -> MeshBoundary:
    """
    `{∂Ωᵢ}ᵢ`
    """
    if verify or complete:
        if mesh.comm.Get_size() > 1:
            raise NotImplementedError('Not supported in parallel.')
        dim = mesh.geometry.dim - 1
        n_boundary_entities = [
            len(locate_entities_boundary(mesh, dim, as_spatial_marker(v)))
            for v in boundaries.values()
        ]
        if verify:
            for i, n in enumerate(n_boundary_entities):
                if n == 0:
                    raise ValueError(f"'{list(boundaries.keys())[i]}' is not on the mesh boundary.")
        if complete:
            marker_all = lambda x: np.full_like(x[0], True)
            n_total = len(locate_entities_boundary(mesh, dim, marker_all))
            if n_total != sum(n_boundary_entities):
                raise ValueError('Boundaries do not cover the complete mesh boundary')
            
    return MeshBoundary(boundaries)