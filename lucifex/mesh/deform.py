from typing import Callable, Iterable

import numpy as np
from dolfinx.mesh import Mesh

from .refine import copy_mesh


def deform_mesh(
    mesh: Mesh,
    clbl: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
    axis: int | None = None,
    bbox: bool = False, 
    name: str | None = None,
) -> Mesh:
    """
    For simplicial or non-simplicial meshes.
    """

    def _deform(x):
        if axis is None:
            return clbl(x)
        else:
            x[:, axis] = clbl(x[:, axis])
            return x

    x = mesh.geometry.x
    x_deformed = _deform(np.copy(x))

    if not len(x_deformed) == len(x):
        raise ValueError('Expected array of equal length after deformation.')
    
    if bbox and not is_same_bbox(x_deformed, x):
        raise ValueError('Expected array of equal bounds after internal deformation.')

    mesh_deformed = copy_mesh(mesh, name)
    mesh_deformed.geometry.x[:, :] = x_deformed
    return mesh_deformed


def stretch_mesh(
    mesh: Mesh,
    clbl: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
    axis: int | None = None,
    name: str | None = None,
) -> Mesh:
    return deform_mesh(mesh, clbl, axis, bbox=False, name=name)
            

def is_same_bbox(
    x_deformed: np.ndarray, 
    x_original: np.ndarray,
    axes: Iterable[int] = (0, 1, 2),
) -> bool:
    for i in axes:
        xo_i = x_original[:, i]
        xd_i = x_deformed[:, i]
        if not np.isclose(np.min(xo_i) , np.min(xd_i)):
            return False
        if not np.isclose(np.max(xo_i), np.max(xd_i)):
            return False
    return True
            
