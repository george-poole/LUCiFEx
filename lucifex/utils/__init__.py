from .deferred import defer
from .fem_utils import (is_scalar, is_vector, is_tensor, extract_mesh, extract_meshes,
                        is_shape, is_continuous_lagrange, is_discontinuous_lagrange, 
                        is_mixed_space, is_same_element, is_family_alias, extract_integrands,
                        extract_integrand)
from .dofs_utils import (
    Marker,
    SpatialMarker, 
    MarkerOrExpression,
    SpatialExpression,
    dofs_indices,
    dofs,
    dofs_transformation,
    extremum,
    maximum,
    minimum,
    as_spatial_marker,
    as_dofs_setter,
    dofs_limits_corrector,
)
from .enum_types import CellType, BoundaryType, DiagonalType
from .numpy_typecast import (
    grid,
    triangulation,
    quadrangulation,
    vertex_to_grid_index_map,
    where_on_grid,
    cross_section,
    spacetime_grid,
    UnstructuredQuadError,
)
from .perturbation import (
    SpatialPerturbation, DofsPerturbation, Perturbation, 
    cubic_noise, sinusoid_noise, rescale,
)
from .measure_utils import create_tagged_measure, integral
from .mesh_utils import(
    mesh_axes,
    mesh_vertices_tensor,
    mesh_vertices,
    mesh_coordinates,
    mesh_axes_spacing,
    cell_sizes,
    cell_size_quantity,
    cell_aspect_ratios,
    n_cells,
    n_entities,
    is_cartesian,
)
from .fem_mutate import (set_fem_constant, set_finite_element_function, 
                           interpolate_finite_element_function, set_value)
from .fem_typecast import finite_element_constant, finite_element_function, function_space, finite_element_function_components
from .numpy_typecast import triangulation, quadrangulation, grid, as_index, as_indices
from .py_utils import (filter_kwargs, log_texec, replicate_callable, ToDoError,
                       optional_lru_cache, MultipleDispatchTypeError, as_slice, StrSlice)
from .norm import L_norm, l_norm, div_norm
