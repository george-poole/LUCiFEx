from .deferred import defer
from .fem_utils import (is_scalar, is_vector, is_tensor, extract_mesh, extract_meshes,
                        is_shape, is_continuous_lagrange, is_discontinuous_lagrange, 
                        is_mixed_space, is_same_element, is_family_alias, extract_integrands,
                        extract_integrand)
from .dofs_utils import (
    SpatialMarkerTypes,
    SpatialMarker, 
    SpatialMarkerOrExpression,
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
from .numpy_typecasting import (
    grid,
    triangulation,
    quadrangulation,
    vertex_to_grid_index_map,
    where_on_grid,
    cross_section,
)
from .fem_perturbation import (
    SpatialPerturbation, DofsPerturbation, Perturbation, 
    cubic_noise, sinusoid_noise, rescale,
)
from .mesh_utils import(
    axes,
    vertices_tensor,
    vertices,
    coordinates,
    axes_spacing,
    cell_sizes,
    cell_size_quantity,
    cell_aspect_ratios,
    n_cells,
    n_entities,
    is_cartesian,
)
from .fem_mutation import (set_fem_constant, set_fem_function, 
                           interpolate_fem_function, set_value)
from .fem_typecasting import fem_constant, fem_function, fem_function_space, fem_function_components
from .numpy_typecasting import triangulation, quadrangulation, grid, as_index, as_indices
from .py_utils import (filter_kwargs, log_texec, replicate_callable, FixMeError,
                       optional_lru_cache, MultipleDispatchTypeError, as_slice, StrSlice)
from .norm import L_norm, l_norm, div_norm
