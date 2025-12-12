from .deferred import defer
from .ufl_utils import (
    is_scalar, 
    is_vector, is_tensor, extract_mesh, extract_meshes,
    is_shape, 
    is_continuous_lagrange, is_discontinuous_lagrange, 
    is_same_element, is_family_alias, extract_integrands,
    extract_integrand,
)
from .dofs_utils import (
    SpatialMarkerAlias,
    SpatialMarker, 
    dofs_indices,
    dofs,
    as_spatial_marker,
)
from .numpy_utils import (
    grid,
    triangulation,
    quadrangulation,
    vertex_to_grid_index_map,
    where_on_grid,
    cross_section,
    spacetime_grid,
    UnstructuredQuadError,
)
from .mesh_utils import(
    CellType,
    DiagonalType,
    BoundaryType,
    create_tagged_measure, 
    mesh_integral,
    mesh_axes,
    mesh_vertices_tensor,
    mesh_vertices,
    mesh_coordinates,
    mesh_axes_spacing,
    cell_sizes,
    cell_size_quantity,
    cell_aspect_ratios,
    number_of_cells,
    number_of_entities,
    is_cartesian,
    is_uniform_cartesian,
)
from .fem_utils import (
    create_fem_constant, 
    create_fem_function, 
    create_fem_space, 
    get_component_fem_functions,
    set_fem_constant, 
    set_fem_function, 
    set_fem_function_interpolate,
    is_mixed_space,
    get_fem_subspace,
    get_fem_subspaces,
)
from .numpy_utils import triangulation, quadrangulation, grid, as_index, as_indices
from .py_utils import (filter_kwargs, log_timing, replicate_callable, ToDoError, nested_dict,
                       optional_lru_cache, MultipleDispatchTypeError, as_slice, StrSlice)
from .str_utils import str_indexed
