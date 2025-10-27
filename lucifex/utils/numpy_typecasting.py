""" Module for converting `dolfinx.mesh.Mesh` and `dolfinx.fem.Function` 
objects to `numpy.ndarray` objects for postprocessing """

from functools import singledispatch
import operator
from typing import overload, Literal, Callable, Iterable

from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from matplotlib.collections import PolyCollection
from matplotlib.tri.triangulation import Triangulation
import numba
import numpy as np

from .enum_types import CellType
from .dofs_utils import dofs
from .py_utils import optional_lru_cache, MultipleDispatchTypeError, StrSlice, as_slice
from .mesh_utils import mesh_vertices, mesh_coordinates, mesh_axes
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

@singledispatch
def _triangulation(arg, *_, **__):
    raise MultipleDispatchTypeError(arg)

@_triangulation.register
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

    x_coordinates, y_coordinates = mesh_coordinates(mesh)

    # coordinates of each vertex in a given triangle cell
    connec = mesh.topology.connectivity(mesh.geometry.dim, 0)
    n_cells = connec.num_nodes
    triangles = [connec.links(i) for i in range(n_cells)]

    trigl = Triangulation(x_coordinates, y_coordinates, triangles)

    return trigl

@_triangulation.register
def _(f: Function):
    """Interpolates function to P₁ (which has identity vertex-to-dof map)
    to evaluate the function at the vertex values.

    Note that this is suitable on both Cartesian and unstructured meshes.
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

    connec = mesh.topology.connectivity(mesh.geometry.dim, 0)
    n_cells = connec.num_nodes
    quads = [reorder_quad_vertices(connec.links(i)) for i in range(n_cells)]

    vertices = xy_coordinates[quads]

    _kwargs = dict(facecolor='white', edgecolor='black')
    _kwargs.update(polycollection_kwargs)
    quadl = PolyCollection(vertices, **_kwargs)

    return quadl


@overload
def _grid(
    mesh: Mesh,
    strict: bool = False,
) -> tuple[np.ndarray, ...]:
    ...

@overload
def _grid(
    f: Function,
    strict: bool = False,
    jit: bool = True,
    use_mesh_map: bool = False,
    use_mesh_cache: bool = True,
    mask: float = np.nan,
) -> np.ndarray:
    ...

@singledispatch
def _grid(arg, *_, **__):
    raise MultipleDispatchTypeError(arg)

@_grid.register
def _(
    mesh: Mesh, 
    strict: bool = False,
):
    return mesh_axes(mesh, strict)

@_grid.register(Function)
def _(
    f: Function,
    strict: bool = False,
    jit: bool = True,
    use_mesh_map: bool = False,
    use_mesh_cache: bool = True,
    mask: float = np.nan,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Interpolates the finite element function to the `P₁` function space
    (which has identity vertex-to-DoF map) to evaluate the function at the vertex values,
    then returns these vertex values in a `np.ndarray`.

    Note that this is suitable only if the mesh is Cartesian.
    """
    if not is_scalar(f):
        raise ScalarError(f)
    
    mesh = f.function_space.mesh
    axes = mesh_axes(use_cache=use_mesh_cache)(mesh, strict)
    vertices = mesh_vertices(mesh)
    vertex_values = dofs(f, ('P', 1), try_identity=True)

    n_vertices = len(vertices)
    n_values = len(vertex_values)
    if n_vertices != n_values:
        raise ValueError(
            f""" 
        Number of vertex values {n_values} does not 
        match number of vertices {n_vertices} """
        )

    if use_mesh_map:
        mapping = vertex_to_grid_index_map(use_cache=use_mesh_cache)(mesh, strict, jit, use_mesh_cache)
        return _grid_from_map(mapping, vertex_values, axes, mask)
    else:
        if jit:
            return _grid_jit(vertex_values, numba.typed.List(vertices), axes, mask)
        else:
            return _grid_nojit(vertex_values, vertices, axes, mask)


@numba.jit(nopython=True)
def _grid_jit(
    vertex_values: np.ndarray,
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    """Returns the vertex values on a Cartesian grid.
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


def _grid_nojit(
    vertex_values: np.ndarray,
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    """
    Returns the vertex values on a Cartesian grid. 
    See also `_grid_jit` for a faster, JIT-compiled version.
    """
    return _grid_jit.__wrapped__(vertex_values, vertices, axes, mask)
    # dim = len(axes)
    # nx = tuple(len(i) for i in axes)
    # f_grid = np.full(nx, mask)

    # for vertex_index, x_vertex in enumerate(vertices):
    #     x_index = [np.where(axes[i] == x_vertex[i])[0][0] for i in range(dim)]
    #     f_grid[tuple(x_index)] = vertex_values[vertex_index]

    # return f_grid


def _grid_from_map(
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


grid = optional_lru_cache(_grid)


@optional_lru_cache
def vertex_to_grid_index_map(
    mesh: Mesh,
    strict: bool = False,
    jit: bool = True,
    use_cache: bool = True,
) -> dict[int, tuple[int, ...]]:
    axes = mesh_axes(use_cache=use_cache)(mesh, strict)
    vertices = mesh_vertices(mesh)
    if jit:
        map_array = _vertex_to_grid_index_map_jit(numba.typed.List(vertices), axes)
    else:
        map_array = _vertex_to_grid_index_map_nojit(vertices, axes)
    return {i: tuple(map_array[i]) for i in range(len(map_array))}


@numba.jit(nopython=True)
def _vertex_to_grid_index_map_jit(
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    n_vertices = len(vertices)
    dim = len(axes)
    map_array = np.empty((n_vertices, dim), dtype=np.int64)

    for vertex_index in range(n_vertices):
        for d in range(dim):
            map_array[vertex_index, d] = np.searchsorted(axes[d], vertices[vertex_index][d])

    return map_array


def _vertex_to_grid_index_map_nojit(
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    return _vertex_to_grid_index_map_jit.__wrapped__(vertices, axes)


def reorder_quad_vertices(
    cell_vertices: np.ndarray,
    initial_ordering: Literal["N", "Z"] = "N",
    final_ordering: Literal["clockwise", "anticlockwise"] = "clockwise",
) -> np.ndarray:
    """
    Quadrilateral meshes generated by `dolfinx` have a local
    indexing of cell vertices that is self-intersecting 
    (e.g. forming an N-shape or Z-shape)

    ```
    1 _______ 3
    |         |
    |         |
    0 ‾‾‾‾‾‾‾ 2
    ```

    A `matplotlib` quadrangulation requires local indexing
    of cell vertices to be non-self-intersecting (i.e. clockwise
    or anticlockwise)

    ```
    1 _______ 2
    |         |
    |         |
    0 ‾‾‾‾‾‾‾ 3
    ```

    """
    assert len(cell_vertices) == 4

    if initial_ordering == 'N':
        if final_ordering == "clockwise":
            initial_final_map = {
                0: 0, 
                1: 1,
                2: 3,
                3: 2,
            }
        elif final_ordering == "anticlockwise":
            initial_final_map = {
                0: 0, 
                1: 2,
                2: 3,
                3: 1,
            }
        else:
            raise ValueError(f'{final_ordering} not recognised')
        
    elif initial_ordering == 'Z':
        raise NotImplementedError
    
    else:
        raise ValueError(f'{initial_ordering} not recognised')
    
    reordered = np.zeros(len(cell_vertices), dtype=np.int32)
    for initial, final in initial_final_map.items():
        reordered[initial] = cell_vertices[final]
    
    return reordered


def where_on_grid(
    f: Function,
    condition: Callable[[np.ndarray], np.ndarray],
    use_cache: tuple[bool, bool] = (True, False),
) -> tuple[np.ndarray, ...]:
    axes = mesh_axes(use_cache=use_cache[0])(f.function_space.mesh)
    f_grid = grid(use_cache=use_cache[1])(f)
    indices = np.where(condition(f_grid))
    return tuple(x[i] for x, i in zip(axes, indices))


def as_index(
    arr: np.ndarray | Iterable[float],
    target: int | float,
    tol: float | None = None,
    less_than: bool | None = None,
) -> int:
    if isinstance(target, int):
        return target
    
    if less_than is None:
        func = None
    else:
        arr = np.sort(arr)
        if less_than:
            func = operator.lt
            index_shift = -1
        else:
            func = operator.ge
            index_shift = 1

    arr_diff = np.abs([i - target for i in arr])
    if tol is not None and np.min(arr_diff) > tol:
        raise ValueError(
            f'No values found within tolerance {tol} of target value {target}.'
        ) 
           
    target_index = np.argmin(arr_diff)

    if func is not None:
        if not func(arr[target_index], target):
            target_index += index_shift
        assert func(arr[target_index], target)

    return target_index


def as_indices(
    arr: np.ndarray | Iterable[float],
    targets: range | list[int | float] | int | StrSlice,
    tol: float | None = None,
    window: bool = False,
) -> Iterable[int]:
    if isinstance(targets, range):
        indices = targets
    elif isinstance(targets, StrSlice):
        slc = as_slice(targets)
        indices = range(slc.start, slc.stop, slc.step)
    elif isinstance(targets, int):
        stop = len(arr)
        step = stop // targets
        indices = range(0, stop, step)
    else:
        indices = [as_index(arr, i, tol) for i in targets]
        if window:
            if indices[0] < targets[0]:
                indices[0] += 1
            if indices[1] > targets[-1]:
                indices[0] -= 1
    return indices


def cross_section(
    fxyz: Function | tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    use_cache: tuple[bool, bool] = (True, False),
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns 
    
    `x_axis, y_line, y_value` in 2D

    `x_axis, y_axis, z_grid, z_value` in 3D
    """
    
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
        axes = grid(use_cache=use_cache[0])(fxyz.function_space.mesh)
        f_grid = grid(use_cache=use_cache[1])(fxyz)
        dim = fxyz.function_space.mesh.geometry.dim
    else:
        f_grid = fxyz[0]
        axes = tuple(fxyz[1:])
        dim = len(axes)

    if dim == 2:
        return _cross_section_line(f_grid, axes, f_fraction, f_value, axis_index)
    elif dim == 3:
        return _cross_section_colormap(f_grid, axes, f_fraction, f_value, axis_index)
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


def spacetime_grid(
    u: Iterable[Function | tuple[np.ndarray, ...]],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    use_cache: tuple[bool, bool] = (True, False),
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> np.ndarray:
    _cross_sections = []
    xaxis, _csec, value  = cross_section(
        u[0], axis, value, fraction, use_cache, axis_names,
    )
    _cross_sections.append(_csec)

    for _u in u[1:]:
        _xaxis, _csec, _value  = cross_section(_u, axis, value, fraction, use_cache, axis_names)
        assert np.isclose(value, _value)
        assert np.all(np.isclose(xaxis, _xaxis))
        _cross_sections.append(_csec)

    return np.array(_cross_sections).T