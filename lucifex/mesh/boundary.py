import numpy as np
from dolfinx.mesh import Mesh, locate_entities_boundary

from ..utils.py_utils import MultiKey
from ..utils.fenicsx_utils import SpatialMarker, SpatialMarkerAlias, as_spatial_marker, ParallelizationError


class MeshBoundary(
    MultiKey[
        str | int,
        SpatialMarker | SpatialMarkerAlias,
    ]
):
    def __init__(
        self, 
        boundaries: dict[str | int, SpatialMarker | SpatialMarkerAlias] | None,
    ):
        if boundaries is None:
            boundaries = {}
        self._boundaries = boundaries

    def _getitem(
        self, 
        key
    ) -> SpatialMarker | SpatialMarkerAlias:
        return self._boundaries[key]
        
    @property
    def markers(self) -> list[SpatialMarker | SpatialMarkerAlias]:
        """
        Markers for all defined sub-boundaries `{∂Ωᵢ}`
        """
        return list(self._boundaries.values())

    @property
    def union(self) -> SpatialMarker:
        """
        Union of defined sub-boundaries `∪ᵢ∂Ωᵢ`
        """
        if self.markers:
            return as_spatial_marker(self.markers)
        else:
            nothing_marker = lambda x: (x[0] > 0) & (x[0] < 0) 
            return nothing_marker

    @property
    def complement(self) -> SpatialMarker:
        """
        Complement of the union of defined sub-boundaries `∂Ω \ ∪ᵢ∂Ωᵢ`

        If the defined boundaries are complete, then `∂Ω = ∪ᵢ∂Ωᵢ` and hence `∂Ω \ ∪ᵢ∂Ωᵢ = ∅`.
        """
        return lambda x: np.logical_not(self.union(x))
    
    @property
    def names(self) -> tuple[str | int, ...]:
        return tuple(self._boundaries.keys())
    
    @property
    def markers(self) -> tuple[SpatialMarker | SpatialMarkerAlias, ...]:
        return tuple(self._boundaries.values())


def mesh_boundary(
    mesh: Mesh,
    boundaries: dict[str | int, SpatialMarker | SpatialMarkerAlias] | None = None,
    verify: bool = True,
    complete: bool = False,
) -> MeshBoundary:
    """
    `{∂Ωᵢ}ᵢ`
    """
    if not boundaries:
        return MeshBoundary(boundaries)
    
    if verify or complete:
        if mesh.comm.Get_size() > 1:
            raise ParallelizationError
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