from scipy.sparse import csr_matrix
from matplotlib.axes import Axes

from ..solver.petsc import PETScMat
from .utils import optional_ax


@optional_ax
def plot_sparsity(
    ax: Axes,
    mat: PETScMat,
    **kwargs
) -> None:
    _kwargs = dict(color='black')
    _kwargs.update(kwargs)
    mat_i, mat_j, mat_v = mat.getValuesCSR()
    mat_csr = csr_matrix((mat_i, mat_j, mat_v))
    ax.spy(mat_csr, **_kwargs)