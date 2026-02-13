import gmsh
from typing import Iterable
from functools import lru_cache

import numpy as np

from ..utils.py_utils import replicate_callable
from ..utils.fenicsx_utils import SpatialMarkerAlias, as_spatial_marker
from .gmsh_utils import create_gmsh_mesh_factory


def mesh_from_splines_model(
    *ccw_splines: Iterable[tuple[float, float]],
    method: str = 'addSpline'
):
    add_spline = getattr(gmsh.model.geo, method)
    cached_point_func = lru_cache(maxsize=None)(
        lambda x, y: gmsh.model.geo.addPoint(x, y, 0.0)
    )
    
    splines = []
    for spln in ccw_splines:
        points = []
        for x, y in spln:
            p = cached_point_func(x, y)
            points.append(p)
        spline = add_spline(points)
        splines.append(spline)

    loop = gmsh.model.geo.addCurveLoop(splines)
    gmsh.model.geo.addPlaneSurface([loop])
    
    return gmsh.model


@replicate_callable(
    create_gmsh_mesh_factory(mesh_from_splines_model, 2, 'mesh_from_boundary_points'),
)
def mesh_from_splines():
    pass


def mesh_from_boundaries_model(
    x_bbox: np.ndarray,
    y_bbox: np.ndarray,
    *ccw_boundaries: SpatialMarkerAlias,
    method: str = 'addSpline',
    rtol: float = 1e-5,
    atol: float = 1e-8 ,    
):
    """
    Creates a 2D mesh from its boundaries, which must be ordered so as to form a closed counter-clockwise loop.
    """
    X_bbox, Y_bbox = np.meshgrid(x_bbox, y_bbox, indexing="ij")

    n_boundaries = len(ccw_boundaries)
    boundaries = [as_spatial_marker(i) for i in ccw_boundaries]
    masks = [bdry((X_bbox, Y_bbox)) for bdry in boundaries]

    intersections = []
    for n in range(n_boundaries):
        msk = masks[n]
        msk_next = masks[(n + 1) % n_boundaries]
        iscn = None
        for i in range(len(x_bbox)):
            for j in range(len(y_bbox)):
                if msk[i, j] and msk_next[i, j]:
                    iscn = (x_bbox[i], y_bbox[j])
            if iscn is not None:
                break
        if iscn is None:
            raise RuntimeError(f'No intersection between boundaries {n} and {n + 1} found.')
        intersections.append(iscn)

    def _is_increasing(
        x1: float, 
        x2: float,
    ) -> bool | None:
        if np.isclose(x1, x2, rtol, atol):
            return None
        else:
            return x2 > x1
        
    def _index_sequence(
        arr: np.ndarray,
        increasing: bool | None,
    ) -> list[int]:
        indices = range(0, len(arr))
        if increasing:
            return list(indices)
        else:
            return list(reversed(indices))

    def _is_in_bounds(
        x: float,
        x1: float,
        x2: float,
        increasing,
    ):
        if increasing is None:
            return True
        if increasing:
            return x >= x1 and x <= x2
        else:
            return x <= x1 and x >= x2   

    ccw_splines = []
    for n in range(n_boundaries):
        msk = masks[n]
        xi_next, yi_next = intersections[n]
        xi_prev, yi_prev = intersections[(n - 1) % n_boundaries]

        x_increasing = _is_increasing(xi_prev, xi_next)
        y_increasing = _is_increasing(yi_prev, yi_next)

        i_sqn = _index_sequence(x_bbox, x_increasing)
        j_sqn= _index_sequence(y_bbox, y_increasing)
        
        ccw_points = []
        for i in i_sqn:
            if not _is_in_bounds(x_bbox[i], xi_prev, xi_next, x_increasing):
                continue
            for j in j_sqn:
                if not _is_in_bounds(y_bbox[j], yi_prev, yi_next, y_increasing):
                    continue
                if msk[i, j]:
                    ccw_points.append((x_bbox[i], y_bbox[j]))
        ccw_splines.append(ccw_points)

    return mesh_from_splines_model(*ccw_splines, method=method)


@replicate_callable(
    create_gmsh_mesh_factory(mesh_from_boundaries_model, 2, 'mesh_from_boundaries'),
)
def mesh_from_boundaries():
    pass