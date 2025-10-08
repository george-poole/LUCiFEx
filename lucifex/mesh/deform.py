from typing import Callable, Iterable

import numpy as np
from dolfinx.mesh import Mesh

from .refine import copy_mesh


def deform(
    mesh: Mesh,
    func: Callable[[np.ndarray], np.ndarray],
    index: int | None = None,
    internal: bool = False, 
    name: str | None = None,
) -> None:
    """
    For simplex meshes only.
    """

    def _deform(x):
        if index is None:
            return func(x)
        else:
            x[:, index] = func(x[:, index])
            return x

    x = mesh.geometry.x
    x_deformed = _deform(np.copy(x))

    if not len(x_deformed) == len(x):
        raise ValueError('Expected array of equal length after deformation.')
    
    if internal and not is_internal_deformation(x_deformed, x):
        raise ValueError('Expected array of equal bounds after internal deformation.')

    mesh_deformed = copy_mesh(mesh, name)
    mesh_deformed.geometry.x[:, :] = x_deformed
    return mesh_deformed
            

def is_internal_deformation(
    x_deformed: np.ndarray, 
    x_original: np.ndarray,
    indices: Iterable[int] = (0, 1, 2),
) -> bool:
    for i in indices:
        xo_i = x_original[:, i]
        xd_i = x_deformed[:, i]
        if not np.isclose(np.min(xo_i) , np.min(xd_i)):
            return False
        if not np.isclose(np.max(xo_i), np.max(xd_i)):
            return False
    return True
            
