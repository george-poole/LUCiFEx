""" Module for converting `dolfinx.mesh.Mesh` and `dolfinx.fem.Function` 
objects to `numpy.ndarray` objects for postprocessing """

from functools import singledispatch
from typing import overload, Literal, Callable, Iterable
from functools import lru_cache

from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from matplotlib.collections import PolyCollection
from matplotlib.tri.triangulation import Triangulation
import numba
import numpy as np

from .enum_types import CellType
from .dofs_utils import dofs
from .py_utils import optional_lru_cache, MultipleDispatchTypeError, StrSlice, as_slice
from .mesh_utils import vertices, coordinates, axes, is_structured
from .fem_utils import is_scalar, ScalarError


@overload
def _triangulation(
    mesh: Mesh,
) -> Triangulation:
    ...

@overload
def _triangulation(
    f: Function,
) -> np.ndarray:
    ...

def _triangulation(obj):
    return __triangulation(obj)


@singledispatch
def __triangulation(obj):
    raise MultipleDispatchTypeError(obj)

@__triangulation.register
def _(mesh: Mesh):
    if not mesh.geometry.dim == 2:
        raise ValueError(
            f"""
        Triangulation only valid in 2D, not 
        dimension {mesh.geometry.dim} """
        )

    if not mesh.topology.cell_name() == CellType.TRIANGLE:
        raise ValueError(
            f""" 
        Triangulation only valid for triangle cells, not 
        {mesh.topology.cell_name()} """
        )

    x_coordinates, y_coordinates = coordinates(mesh)

    # coordinates of each vertex in a given triangle cell
    connec = mesh.topology.connectivity(mesh.geometry.dim, 0)
    n_cells = connec.num_nodes
    triangles = [connec.links(i) for i in range(n_cells)]

    trigl = Triangulation(x_coordinates, y_coordinates, triangles)

    return trigl

@__triangulation.register
def _(f: Function):
    """Interpolates function to P₁ (which has identity vertex-to-dof map)
    to evaluate the function at the vertex values.

    Note that this is suitable on both structured and unstructured meshes.
    """
    return dofs(f, ('P', 1), try_identity=True)


triangulation = optional_lru_cache(_triangulation)


@optional_lru_cache
def quadrangulation(
    mesh: Mesh,
    **polycollection_kwargs,
) -> PolyCollection:
    if not mesh.geometry.dim == 2:
        raise ValueError(
            f"""
        Quadrangulation only valid in 2D, not 
        dimension {mesh.geometry.dim} """
        )

    if not mesh.topology.cell_name() == CellType.QUADRILATERAL:
        raise ValueError(
            f""" 
        Quadrangulation only valid with quadrilateral cells, not 
        {mesh.topology.cell_name()} """
        )

    xy_coordinates = mesh.geometry.x[:, :2]

    # coordinates of each vertex in a given quadrilateral cell
    connec = mesh.topology.connectivity(mesh.geometry.dim, 0)
    n_cells = connec.num_nodes
    quads = [reorder_local_vertices(connec.links(i)) for i in range(n_cells)]

    vertices = xy_coordinates[quads]

    _kwargs = dict(facecolor='white', edgecolor='black')
    _kwargs.update(polycollection_kwargs)
    quadl = PolyCollection(vertices, **_kwargs)

    return quadl



@overload
def _grid(
    f: Function,
    strict: bool = False,
    jit: bool | None = None,
) -> np.ndarray:
    ...


@overload
def _grid(
    mesh: Mesh,
    strict: bool = False,
) -> tuple[np.ndarray, ...]:
    ...

def _grid(*args ,**kwargs):
    return __grid(*args, **kwargs)


@singledispatch
def __grid(obj):
    raise MultipleDispatchTypeError(obj)


@__grid.register
def _(
    mesh: Mesh, 
    strict: bool = False,
):
    return axes(mesh, strict)


@__grid.register(Function)
def _(
    f: Function,
    strict: bool = False,
    jit: bool | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Interpolates the finite element function to the `P₁` function space
    (which has identity vertex-to-DoF map) to evaluate the function at the vertex values,
    then returns these vertex values in a `np.ndarray`.

    Note that this is suitable only if the mesh is structured.
    """
    if not is_scalar(f):
        raise ScalarError(f)
    
    mesh = f.function_space.mesh
    x_axes = grid(use_cache=True)(mesh, strict)
    f_vertices = dofs(f, ('P', 1), try_identity=True)

    if jit is True:
        _mesh_func = lambda l: numba.typed.List(vertices(l))
        _grid_func = _grid_jit
    elif jit is False:
        _mesh_func = vertices
        _grid_func = _grid_nojit
    elif jit is None:
        _mesh_func = vertex_to_grid_index_map
        _grid_func = _grid_from_map
    else:
        raise ValueError

    vertices_or_map = _mesh_func(mesh)
    if len(f_vertices) != len(vertices_or_map):
        raise ValueError(
            f""" 
        Number of vertex values {vertices_or_map} does not 
        match number of points {f_vertices} """
        )
    return _grid_func(f_vertices, vertices_or_map, x_axes)


@numba.jit(nopython=True)
def _grid_jit(
    f_vertices: np.ndarray,
    x_vertices: list[tuple[float, ...]],
    x_axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    """A JIT-compiled function that returns the vertex values on a
    structured grid. See also `_function_grid_no_jit` for a slower, non-compiled version.
    """

    n_vertex = len(f_vertices)
    dim = len(x_axes)

    match dim:
        case 1:
            nx = len(x_axes[0])
            f_grid = np.zeros(nx)
            for vertex_index in range(n_vertex):
                x_index = np.where(x_axes[0] == x_vertices[vertex_index][0])[0][0]
                f_grid[x_index] = f_vertices[vertex_index]
        case 2:
            nx, ny = len(x_axes[0]), len(x_axes[1])
            f_grid = np.zeros((nx, ny))
            for vertex_index in range(n_vertex):
                x_index = np.where(x_axes[0] == x_vertices[vertex_index][0])[0][0]
                y_index = np.where(x_axes[1] == x_vertices[vertex_index][1])[0][0]
                f_grid[x_index, y_index] = f_vertices[vertex_index]
        case 3:
            nx, ny, nz = len(x_axes[0]), len(x_axes[1]), len(x_axes[2])
            f_grid = np.zeros((nx, ny, nz))
            for vertex_index in range(n_vertex):
                x_index = np.where(x_axes[0] == x_vertices[vertex_index][0])[0][0]
                y_index = np.where(x_axes[1] == x_vertices[vertex_index][1])[0][0]
                z_index = np.where(x_axes[2] == x_vertices[vertex_index][2])[0][0]
                f_grid[x_index, y_index, z_index] = f_vertices[vertex_index]
        case _:
            raise ValueError(f"Spatial dimension should be 1, 2 or 3, not {dim}")

    return f_grid


def _grid_nojit(
    f_vertices: np.ndarray,
    x_vertices: list[tuple[float, ...]],
    x_axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    """A function that returns the vertex values on a structured
    grid.  See also `_function_grid_jit` for a faster, JIT-compiled version."""
    dim = len(x_axes)
    nx = [len(i) for i in x_axes]
    f_grid = np.zeros(nx)

    for vertex_index, x_vertex in enumerate(x_vertices):
        x_index = [np.where(x_axes[i] == x_vertex[i])[0][0] for i in range(dim)]
        f_grid[tuple(x_index)] = f_vertices[vertex_index]

    return f_grid


def _grid_from_map(
    f_vertices: np.ndarray,
    vertex_grid_map: dict[int, int | tuple[int, ...]],
    x_axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    nx = tuple(len(i) for i in x_axes)
    f_grid = np.zeros(nx)

    for vertex_index, x_index in vertex_grid_map.items():
        f_grid[x_index] = f_vertices[vertex_index]

    return f_grid


grid = optional_lru_cache(_grid)


def vertex_to_grid_index_map(
    mesh: Mesh,
    jit: bool = True,
) -> dict[int, int | tuple[int, ...]]:
    """LRU-cached and JIT-compiled under the hood. JIT-compilation
    requires a `numba` installation."""
    if not is_structured(mesh):
        raise ValueError("Mesh must be structured")
    return _vertex_to_grid_index_map(mesh, jit)


@lru_cache
def _vertex_to_grid_index_map(
    mesh: Mesh, 
    jit: bool,
) -> dict[int, int | tuple[int, ...]]:
    grid_vertices = vertices(mesh)
    grid_axes = grid(use_cache=True)(mesh)
    if jit:
        return _vertex_to_grid_index_map_jit(numba.typed.List(grid_vertices), grid_axes)
    else:
        return _vertex_to_grid_index_map_nojit(grid_vertices, grid_axes)


@numba.jit(nopython=True)
def _vertex_to_grid_index_map_jit(
    grid_vertices, 
    grid_axes,
) -> dict[int, tuple[int, ...]]:
    n_vertices = len(grid_vertices)
    dim = len(grid_axes)
    vertex_mapping = {}

    match dim:
        case 1:
            for vertex_index in range(n_vertices):
                x_index = np.where(grid_axes[0] == grid_vertices[vertex_index][0])[0][0]
                vertex_mapping[vertex_index] = (x_index,)
        case 2:
            for vertex_index in range(n_vertices):
                x_index = np.where(grid_axes[0] == grid_vertices[vertex_index][0])[0][0]
                y_index = np.where(grid_axes[1] == grid_vertices[vertex_index][1])[0][0]
                vertex_mapping[vertex_index] = (x_index, y_index)
        case 3:
            for vertex_index in range(n_vertices):
                x_index = np.where(grid_axes[0] == grid_vertices[vertex_index][0])[0][0]
                y_index = np.where(grid_axes[1] == grid_vertices[vertex_index][1])[0][0]
                z_index = np.where(grid_axes[2] == grid_vertices[vertex_index][2])[0][0]
                vertex_mapping[vertex_index] = (x_index, y_index, z_index)
        case _:
            raise ValueError(f"Spatial dimension should be 1, 2 or 3, not {dim}")

    return vertex_mapping


def _vertex_to_grid_index_map_nojit(
    grid_vertices: list[tuple[float, ...]],
    grid_axes: tuple[np.ndarray, ...],
) -> dict[int, tuple[int, ...]]:
    dim = len(grid_axes)
    vertex_mapping = {}

    for vertex_index, x_vertex in enumerate(grid_vertices):
        x_index = [np.where(grid_axes[i] == x_vertex[i])[0][0] for i in range(dim)]
        vertex_mapping[vertex_index] = tuple(x_index)

    return vertex_mapping


# TODO what about gmsh meshes?
def reorder_local_vertices(
    cell_vertices: np.ndarray,
    start: Literal["N", "Z"] = "N",
    finish: Literal["clockwise", "anticlockwise"] = "clockwise",
) -> np.ndarray:
    """ "
    Quadrilateral meshes generated by `dolfinx` have a local
    indexing of cell vertices forming an N-shape

    ```
    1 _______ 3
    |         |
    |         |
    0 ‾‾‾‾‾‾‾ 2
    ```

    A quadrangulation requires vertices to have a local
    indexing that is non-self-intersecting (i.e. clockwise
    or anticlockwise)

    ```
    1 _______ 2
    |         |
    |         |
    0 ‾‾‾‾‾‾‾ 3
    ```

    """
    assert len(cell_vertices) == 4
    assert start == "N"

    reordered = np.zeros(len(cell_vertices), dtype=np.int32)
    if finish == "clockwise":
        reordered[0] = cell_vertices[0]
        reordered[1] = cell_vertices[1]
        reordered[2] = cell_vertices[3]
        reordered[3] = cell_vertices[2]
    elif finish == "anticlockwise":
        reordered[0] = cell_vertices[0]
        reordered[1] = cell_vertices[2]
        reordered[2] = cell_vertices[3]
        reordered[3] = cell_vertices[1]
    else:
        raise ValueError

    return reordered


def where_on_grid(
    f: Function,
    condition: Callable[[np.ndarray], np.ndarray],
    use_cache: bool = False,
) -> tuple[np.ndarray, ...]:
    f_grid = grid(use_cache=use_cache)(f)
    axes = grid(use_cache=True)(f.function_space.mesh)
    indices = np.where(condition(f_grid))
    return tuple(x[i] for x, i in zip(axes, indices))


def as_index(
    u: np.ndarray | Iterable[float],
    i: int | float,
    tol: float | None = None,
) -> int:
    if isinstance(i, int):
        return i
    else:
        u_diff = np.abs([ui - i for ui in u])
        if tol is not None:
            if np.min(u_diff) > tol:
                raise ValueError(
                    f'No values found within tolerance {tol} of target value {i}.'
                )
        return np.argmin(u_diff)


def as_indices(
    u: np.ndarray | Iterable[float],
    i: range | list[int | float] | int | StrSlice,
    tol: float | None = None,
    window: bool = False,
) -> Iterable[int]:
    if isinstance(i, range):
        indices = i
    elif isinstance(i, StrSlice):
        slc = as_slice(i)
        indices = range(slc.start, slc.stop, slc.step)
    elif isinstance(i, int):
        stop = len(u)
        step = stop // i
        indices = range(0, stop, step)
    else:
        indices = [as_index(u, i, tol) for i in i]
        if window:
            if indices[0] < i[0]:
                indices[0] += 1
            if indices[1] > i[-1]:
                indices[0] -= 1
    return indices


def cross_section(
    fxyz: Function | tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    use_cache: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    
    if fraction:
        f_fraction = value
        if f_fraction < 0 or f_fraction > 1:
            raise ValueError("Fraction must be in interval [0, 1]")
        f_value = None
    else:
        f_fraction = None
        f_value = value

    if not isinstance(axis, int):
        axis_index = axis_names.index(axis)
    else:
        axis_index = axis

    if isinstance(fxyz, Function):
        x = grid(use_cache=True)(fxyz.function_space.mesh)
        f_grid = grid(use_cache=use_cache)(fxyz)
        dim = fxyz.function_space.mesh.geometry.dim
    else:
        f_grid = fxyz[0]
        x = fxyz[1:]
        dim = len(x)

    if dim == 2:
        return _cross_section_line(f_grid, x, f_fraction, f_value, axis_index)
    elif dim == 3:
        return _cross_section_colormap(f_grid, x, f_fraction, f_value, axis_index)
    else:
        raise ValueError(f'Cannot get a cross-section in d={dim}.')


def _cross_section_line(
    f_grid: np.ndarray,
    xy: tuple[np.ndarray, np.ndarray],
    y_fraction: float | None,
    y_value: float | int | None,
    y_index: Literal[0, 1],
) -> tuple[np.ndarray, np.ndarray, float]:
    
    y_axis = xy[y_index]
    x_axis = xy[(y_index + 1) % 2]

    if y_value is not None:
        yaxis_index = as_index(y_axis, y_value)
    else:
        assert y_fraction is not None
        if np.isclose(y_fraction, 1):
            yaxis_index = -1
        else:
            yaxis_index = int(y_fraction * len(y_axis))
    y_value = y_axis[yaxis_index]

    if y_index == 0:
        y_line = f_grid[yaxis_index, :]
    elif y_index == 1:
        y_line = f_grid[:, yaxis_index]
    else:
        raise ValueError
    
    return x_axis, y_line, y_value


def _cross_section_colormap(
    f_grid: np.ndarray,
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    z_fraction: float | None,
    z_value: float | int | None,
    z_index: Literal[0, 1, 2],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    
    z_axis = xyz[z_index]
    x_axis = xyz[(z_index + 1) % 3]
    y_axis = xyz[(z_index + 2) % 3]

    if z_value is not None:
        zaxis_index = as_index(z_axis, z_value)
    else:
        assert z_fraction is not None
        zaxis_index = int(z_fraction * len(z_axis))
    z_value = z_axis[zaxis_index]

    if z_index == 0:
        z_grid = f_grid[zaxis_index, :, :]
    elif z_index == 1:
        z_grid = f_grid[:, zaxis_index, :]
    elif z_index == 2:
        z_grid = f_grid[:, :, zaxis_index]
    else:
        raise ValueError

    return x_axis, y_axis, z_grid, z_value