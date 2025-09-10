from typing import overload

import numpy as np
from dolfinx.mesh import Mesh, locate_entities_boundary

from ..utils.dofs_utils import SpatialMarkerFunc, as_spatial_indicator_func


class MeshBoundary:
    def __init__(self, boundaries: dict[str | int, SpatialMarkerFunc]):
        self._boundaries = boundaries

    @overload
    def __getitem__(
        self, 
        tag: str | int,
    ) -> SpatialMarkerFunc:
        ...

    @overload
    def __getitem__(
        self, 
        tag: tuple[str | int, ...],
    ) -> list[SpatialMarkerFunc]:
        ...

    def __getitem__(
        self, 
        tag: str | int | tuple[str | int, ...],
    ) -> SpatialMarkerFunc | list[SpatialMarkerFunc]:
        if isinstance(tag, tuple):
            return [self._boundaries[t] for t in tag]
        else:
            return self._boundaries[tag]

    @property
    def union(self) -> list[SpatialMarkerFunc]:
        """
        `∪ᵢ ∂Ωᵢ`
        """
        return list(self._boundaries.values())
    
    @property
    def names(self) -> tuple[str | int, ...]:
        return tuple(self._boundaries.keys())
    
    @property
    def markers(self) -> tuple[SpatialMarkerFunc, ...]:
        return tuple(self._boundaries.values())


def mesh_boundary(
    mesh: Mesh,
    boundaries: dict[str | int, SpatialMarkerFunc],
    verify: bool = True,
    complete: bool = False,
) -> MeshBoundary:
    """
    `{∂Ωᵢ}ᵢ`
    """
    
    if verify or complete:
        dim = mesh.geometry.dim - 1
        if mesh.comm.Get_size() > 1:
            raise NotImplementedError('Not supported in parallel.')
        n_boundary_entities = [
            len(locate_entities_boundary(mesh, dim, as_spatial_indicator_func(v)))
            for v in boundaries.values()
        ]
        if verify:
            for i, n in enumerate(n_boundary_entities):
                if n == 0:
                    raise ValueError(f"'{list(boundaries.keys())[i]}' is not on the mesh boundary.")
        if complete:
            marker_all = lambda x: np.full_like(x[0], True)
            n_total = locate_entities_boundary(mesh, dim, marker_all)
            if n_total != sum(n_boundary_entities):
                raise ValueError('Boundaries do not cover the complete mesh boundary')
            
    return MeshBoundary(boundaries)