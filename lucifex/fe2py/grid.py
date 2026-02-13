from typing_extensions import Self

from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
import numba
import numpy as np

from ..fdm import FunctionSeries
from ..utils.fenicsx_utils import (
    create_function, dofs,
    mesh_vertices, mesh_axes,
    is_scalar, ScalarError,
)
from ..utils.py_utils import optional_lru_cache, replicate_callable
from .abc import FE2PySeries


class GridMesh:
    def __init__(
        self, 
        axes: tuple[np.ndarray, ...],
        name: str | None = None,
    ):
        self._axes = axes
        self._name = name

    @classmethod
    def from_mesh(
        cls: type['GridMesh'], 
        mesh: Mesh,
        strict: bool = False,
    ) -> Self:
        axes = mesh_axes(use_cache=True)(mesh, strict)
        name = mesh.name
        return cls(axes, name)
    
    @property
    def x(self) -> np.ndarray:
        return self._get_axis(0)
    
    @property
    def y(self) -> np.ndarray | None:
        return self._get_axis(1)
    
    @property
    def z(self) -> np.ndarray | None:
        return self._get_axis(2)
        
    def _get_axis(self, index: int):
        try:
            return self._axes[index]
        except IndexError:
            return None 

    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        return self._axes

    @property
    def name(self) -> str | None:
        return self._name
    

as_grid_mesh = optional_lru_cache(replicate_callable(GridMesh.from_mesh)(lambda: None))


class GridFunction:

    def __init__(
        self, 
        values: np.ndarray,
        axes: tuple[np.ndarray, ...],
        name: str | None = None,
    ):
        self._values = values
        self._grid = GridMesh(axes)
        self._name = name

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def grid(self) -> GridMesh:
        return self._grid

    @property
    def name(self) -> str | None:
        return self._name

    @classmethod
    def from_function(
        cls: type['GridFunction'],
        u: Function,
        strict: bool = False,
        jit: bool = True,
        use_mesh_map: bool = False,
        use_mesh_cache: bool = True,
        mask: float = np.nan,
    ) -> Self:
        """Interpolates the finite element function to the `P₁` function space
        (which has identity vertex-to-DoF map) to evaluate the function at the vertex values,
        then returns these vertex values in a `np.ndarray`.

        Note that this is suitable only if the mesh is Cartesian.
        """
        values = grid_dofs(u, strict, jit, use_mesh_map, use_mesh_cache, mask)
        grid_mesh = as_grid_mesh(use_cache=True)(u.function_space.mesh, strict)
        return cls(values, grid_mesh.axes, u.name)
    

as_grid_function = optional_lru_cache(replicate_callable(GridFunction.from_function)(lambda: None))


class GridSeries(
    FE2PySeries[GridMesh, GridFunction]
):
    def __init__(
        self, 
        series: list[np.ndarray], 
        t: list[float], 
        axes: tuple[np.ndarray, ...],
        name: str | None = None,
    ): 
        super().__init__(series, t, name)
        self._axes = axes
    
    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        return self._axes
    
    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self:
        if name is None:
            name = self._create_subname(index)
        return GridSeries([i[index] for i in self.series], self.time_series, self.axes, name)
    
    @classmethod
    def from_series(
        cls: type['GridSeries'], 
        u: FunctionSeries,
        use_cache: tuple[bool, bool] = (True, True),
        slc: slice = slice(None, None, None),
        **grid_kwargs,
    ):
        grid = ...
        series, time_series, axes = super().from_series(
            grid,
            u,
            use_cache,
            slc,
            **grid_kwargs,
        )
        return cls(series, time_series, axes, u.name)


def grid_dofs(
    f: Function | Expr,
    strict: bool = False,
    jit: bool = True,
    use_mesh_map: bool = False,
    use_mesh_cache: bool = True,
    mask: float = np.nan,
) -> np.ndarray:
    """Interpolates the finite element function to the `P₁` function space
    (which has identity vertex-to-DoF map) to evaluate the function at the vertex values,
    then returns these vertex values in a `np.ndarray`.

    Note that this is suitable only if the mesh is Cartesian.
    """
    if not is_scalar(f):
        raise ScalarError(f)
    
    if not isinstance(f, Function):
        f = create_function((mesh, 'P', 1), f)
    
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
        mapping = vertex_to_grid_index_map(use_cache=use_mesh_cache)(
            mesh, strict, jit, use_mesh_cache
        )
        return _grid_dofs_from_map(mapping, vertex_values, axes, mask)
    else:
        if jit:
            return _grid_dofs_jit(vertex_values, numba.typed.List(vertices), axes, mask)
        else:
            return _grid_dofs_nojit(vertex_values, vertices, axes, mask)


@numba.jit(nopython=True)
def _grid_dofs_jit(
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


def _grid_dofs_nojit(
    vertex_values: np.ndarray,
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    """
    Returns the vertex values on a Cartesian grid. 
    See also `_grid_jit` for a faster, JIT-compiled version.
    """
    return _grid_dofs_jit.__wrapped__(vertex_values, vertices, axes, mask)


def _grid_dofs_from_map(
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
     