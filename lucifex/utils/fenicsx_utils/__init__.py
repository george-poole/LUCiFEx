from .dofs_grid import dofs_grid
from .dofs_utils import (
    Marker,
    MarkerAlias,
    BooleanMarker, 
    dofs_indices,
    dofs,
    as_boolean_marker,
    limits_corrector,
    DofsLocatorType,
    FacetLocatorType,
)
from .expr_utils import (
    is_scalar, 
    is_vector, is_tensor, extract_mesh, extract_meshes,
    is_shape, 
    NonVectorError,
    NonScalarError,
    NonScalarVectorError,
    ShapeError,
    MeshExtractionError,
    Scaled,
    extract_function_space,
)
from .elem_utils import (
    is_continuous_lagrange, is_discontinuous_lagrange, 
    is_equivalent_element, is_equivalent_family,
)
from .form_utils import (
    extract_integrands,
    extract_integrand,
    is_none,
    is_scaled_type,
    BlockForm,
    extract_bilinear_form,
    extract_linear_form,
    create_zero_form,
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
    is_grid,
    is_uniform_grid,
    is_simplicial,
    QuadNonGridMeshError,
    NonGridMeshError,
    ParallelizationError,
    NonSimplexMeshError,
)
from .constant_utils import (
    create_constant, 
    set_constant,
)
from .function_utils import (
    as_function,
    create_function, 
    set_function, 
    set_function_interpolate,
    set_function_dofs,
    extract_component_functions,
    extract_subfunctions,
)
from .function_space_utils import (
    create_function_space,     
    is_mixed_space,
    is_component_space,
    extract_subspace,
    extract_subspaces,
)