import numpy as np
import numba

from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function

from ..py_utils import optional_lru_cache
from .dofs_utils import dofs
from .function_utils import create_function
from .mesh_utils import mesh_vertices, mesh_axes
from .expr_utils import is_scalar, NonScalarError


def dofs_grid(
    f: Function | Expr,
    elem: tuple[str, int] = ('P', 1),
    strict: bool = False,
    jit: bool = True,
    mask: float = np.nan,
    use_mapping: bool = True,
    use_cache: bool = True,
    use_dof_coordinates: bool = False,
    mesh: Mesh | None = None,
) -> np.ndarray:
    """
    Interpolates the finite element function to the `P₁` function space
    to evaluate the function at the vertex values, then returns these 
    vertex values as a grid-like array.
    """
    if not is_scalar(f):
        raise NonScalarError(f)
    
    if not isinstance(f, Function):
        if mesh is not None:
            elem = (mesh, *elem)
        f = create_function(elem, f)

    if use_dof_coordinates:
        vertex_values = dofs(f, create=False)
    else:
        vertex_values = dofs(f, ('P', 1), create=False)

    f_or_mesh = f if use_dof_coordinates else f.function_space.mesh
    axes, vertices = _axes_vertices(use_cache=use_cache)(
        f_or_mesh, strict, use_cache, use_dof_coordinates,
    )

    n_vertices = len(vertices)
    n_values = len(vertex_values)
    if n_vertices != n_values:
        raise ValueError(
            f""" 
        Number of vertex values {n_values} does not 
        match number of vertices {n_vertices} """
        )

    if use_mapping:
        mapping = vertex_to_grid_index_mapping(use_cache=use_cache)(
            f_or_mesh, strict, jit, use_cache, use_dof_coordinates,
        )
        return _dofs_grid_from_mapping(mapping, vertex_values, axes, mask)
    else:
        if jit:
            return _dofs_grid_jit(vertex_values, numba.typed.List(vertices), axes, mask)
        else:
            return _dofs_grid_nojit(vertex_values, vertices, axes, mask)


@optional_lru_cache
def _axes_vertices(
    arg: Mesh | Function,
    strict: bool,
    use_cache: bool,
    use_dof_coordinates: bool,
):
    if use_dof_coordinates:
        if not isinstance(arg, Function):
            raise TypeError('`Function` type required if `use_dof_coordinates = True`')
        f = arg
        mesh = arg.function_space.mesh
        dim = mesh.geometry.dim
        tabulated = f.function_space.tabulate_dof_coordinates()
        dof_coordinates = tuple(tabulated[:, i] for i in range(dim))
        axes = tuple(np.sort(np.unique(i)) for i in dof_coordinates)
        vertices = [tuple(i[:dim]) for i in tabulated]
    else:
        if not isinstance(arg, Mesh):
            mesh = arg.function_space.mesh
        else:
            mesh = arg
        axes = mesh_axes(use_cache=use_cache)(mesh, strict)
        vertices = mesh_vertices(mesh)
    return axes, vertices


@numba.jit(nopython=True)
def _dofs_grid_jit(
    vertex_values: np.ndarray,
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    """
    Returns the vertex values on a Cartesian grid.
    See also `_grid_nojit` for a slower, non-compiled version.
    """

    n_vertex = len(vertex_values)
    dim = len(axes)

    if dim ==1 :
        nx = len(axes[0])
        f_grid = np.full((nx, ), mask)
        for vertex_index in range(n_vertex):
            x_index = np.searchsorted(axes[0], vertices[vertex_index][0])
            f_grid[x_index] = vertex_values[vertex_index]
    elif dim == 2:
            nx, ny = len(axes[0]), len(axes[1])
            f_grid = np.full((nx, ny), mask)
            for vertex_index in range(n_vertex):
                x_index = np.searchsorted(axes[0], vertices[vertex_index][0])
                y_index = np.searchsorted(axes[1], vertices[vertex_index][1])
                f_grid[x_index, y_index] = vertex_values[vertex_index]
    elif dim == 3:
        nx, ny, nz = len(axes[0]), len(axes[1]), len(axes[2])
        f_grid = np.full((nx, ny, nz), mask)
        for vertex_index in range(n_vertex):
            x_index = np.searchsorted(axes[0], vertices[vertex_index][0])
            y_index = np.searchsorted(axes[1], vertices[vertex_index][1])
            z_index = np.searchsorted(axes[2], vertices[vertex_index][2])
            f_grid[x_index, y_index, z_index] = vertex_values[vertex_index]
    else:
        raise ValueError(f"Spatial dimension should be 1, 2 or 3, not {dim}")

    return f_grid


def _dofs_grid_nojit(
    vertex_values: np.ndarray,
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    """
    Returns the vertex values on a Cartesian grid. 
    See also `_grid_jit` for a faster, JIT-compiled version.
    """
    return _dofs_grid_jit.__wrapped__(vertex_values, vertices, axes, mask)


def _dofs_grid_from_mapping(
    vertex_grid_map: np.ndarray,
    vertex_values: np.ndarray,
    x_axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    nx = tuple(len(i) for i in x_axes)
    f_grid = np.full(nx, mask)

    for vertex_index, x_index in vertex_grid_map.items():
        f_grid[x_index] = vertex_values[vertex_index]

    return f_grid


@optional_lru_cache
def vertex_to_grid_index_mapping(
    arg: Mesh | Function,
    strict: bool = False,
    jit: bool = True,
    use_cache: bool = True,
    use_dof_coordinates: bool = False,
) -> dict[int, tuple[int, ...]]:
    axes, vertices = _axes_vertices(use_cache=use_cache)(
        arg, strict, use_cache, use_dof_coordinates,
    )
    if jit:
        mapping = _vertex_to_grid_index_mapping_jit(numba.typed.List(vertices), axes)
    else:
        mapping = _vertex_to_grid_index_mapping_nojit(vertices, axes)
    return {i: tuple(mapping[i]) for i in range(len(mapping))}


@numba.jit(nopython=True)
def _vertex_to_grid_index_mapping_jit(
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    n_vertices = len(vertices)
    dim = len(axes)
    mapping = np.empty((n_vertices, dim), dtype=np.int64)

    for vertex_index in range(n_vertices):
        for d in range(dim):
            mapping[vertex_index, d] = np.searchsorted(axes[d], vertices[vertex_index][d])

    return mapping


def _vertex_to_grid_index_mapping_nojit(
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    return _vertex_to_grid_index_mapping_jit.__wrapped__(vertices, axes)